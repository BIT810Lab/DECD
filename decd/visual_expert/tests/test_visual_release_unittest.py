"""Bounded Visual-only acceptance: no network, CUDA, private data or full100/50.

Run from root: python -B -m unittest discover -s decd/visual_expert/tests
              -p test_visual_release_unittest.py -v
"""
from __future__ import annotations

import copy
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1',
                  TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader
torch.set_num_threads(1)

from decd.visual_expert import model, data, losses, risk, checkpoint
from decd.visual_expert import manifests as mf
from decd.visual_expert.sampler import DomainLabelBalancedBatchSampler, EXPECTED_CELLS

BENCH = runpy.run_path(str(ROOT/'tools/build_mosaic_benchmark.py'))
CONVERT = runpy.run_path(str(ROOT/'tools/convert_mosaic_manifest.py'))
PACKAGE = runpy.run_path(str(ROOT/'tools/package_visual_release.py'))


def csv_write(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    return path


class TinyVisual(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.output_dim = 3
        self.transformer = torch.nn.Module()
        self.transformer.resblocks = torch.nn.ModuleList([torch.nn.Linear(3, 3) for _ in range(4)])
        self.ln_post = torch.nn.LayerNorm(3)


class TinyClip(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.visual = TinyVisual()
        self.frozen_text_placeholder = torch.nn.Parameter(torch.ones(1))

    def encode_image(self, images, normalize=False):
        x = images.mean((2, 3))
        for block in self.visual.transformer.resblocks: x = torch.tanh(block(x))
        return self.visual.ln_post(x)


def factory(**kwargs):
    detector = model.CLIPBinaryDetector(TinyClip(), 3)
    info = model.configure_trainable_parameters(detector, 2); info['feature_dim'] = 3
    return detector, info


def invoke(script, args):
    with patch.object(sys, 'argv', [script] + list(map(str, args))):
        return runpy.run_path(str(ROOT/'scripts'/script), run_name='__main__')


def virtual_pool(n=1003):
    rows = []
    for domain in mf.DOMAIN_TO_ID:
        for label in (0, 1):
            for i in range(n):
                sid = f'{domain}_{label}_{i:05d}'; sha = hashlib.sha256(sid.encode()).hexdigest()
                rows.append(dict(sample_id=sid, image_path=Path('unused')/(sid+'.png'),
                    domain=domain, label=label, label_name='Real' if label==0 else 'Fake',
                    sha256=sha, original_sha256=sha, group_id=''))
    return rows


class Fixture(unittest.TestCase):
    def setUp(self):
        base = Path(os.environ.get('DECD_RELEASE_TEST_TMP', tempfile.gettempdir())).resolve()
        base.mkdir(parents=True, exist_ok=True)
        temp = tempfile.TemporaryDirectory(prefix='visual-code-release-', dir=base)
        self.p = Path(temp.name).resolve(); self.assertTrue(self.p.is_relative_to(base))
        self.addCleanup(temp.cleanup)
        self.cfg = yaml.safe_load((ROOT/'configs/visual_expert_reported.yaml').read_text())
        self.rows = {}; self.paths = {}
        for split, blue in [('train', 31), ('dev', 177)]:
            rows = []
            for d, domain in enumerate(mf.DOMAIN_TO_ID):
                for y in (0, 1):
                    image = self.p/f'{split}_{d}_{y}.png'
                    Image.new('RGB', (9, 7), (d*70, y*150, blue)).save(image)
                    rows.append(dict(sample_id=image.stem, image_path=image.name,
                        domain=domain, label=y, sha256=mf.file_sha256(image),
                        split=split, group_id=image.stem))
            self.rows[split] = rows; self.paths[split] = csv_write(self.p/f'{split}.csv', rows)

    def parent_fixture(self):
        """Mock protocol endpoint, not100 real updates or an author checkpoint."""
        detector, info = factory()
        optimizer = model.build_optimizer(info, backbone_lr=self.cfg['backbone_lr'],
            head_lr=self.cfg['head_lr'], weight_decay=self.cfg['weight_decay'])
        detector(torch.ones(2, 3, 2, 2))[0].sum().backward(); optimizer.step()
        for state in optimizer.state.values(): state['step'].fill_(100)
        rows = data.load_manifest(self.paths['train'], role='train')
        path = self.p/'MOCK_NOT_TRAINED_B100.pth'
        checkpoint.save_checkpoint(path, detector=detector, optimizer=optimizer, config=self.cfg,
            feature_dim=3, metadata=dict(stage='reference', completed_batches=100,
                synthetic_endpoint_fixture=True, train_manifest_sha256=mf.file_sha256(self.paths['train']),
                train_content_sha256=mf.content_sha256(rows)))
        return path, detector, optimizer


class InitializationContract(Fixture):
    """Tiny CPU fixtures only; never real OpenCLIP weight validation."""

    def local(self, payload=None, name='raw_clip.pt'):
        path=self.p/name
        torch.save(TinyClip().state_dict() if payload is None else payload,path)
        return path

    def build(self,path,create=None,**extra):
        create=create or unittest.mock.Mock(return_value=TinyClip())
        fake=types.SimpleNamespace(create_model=create,__version__='CPU-MOCK-NOT-REAL')
        with patch.dict(sys.modules,{'open_clip':fake}), patch.object(model,'version',return_value='CPU-MOCK-NOT-REAL'):
            return model.build_detector(model_name=self.cfg['model_name'],pretrained='openai',
                image_size=224,trainable_blocks=2,device=torch.device('cpu'),
                pretrained_path=path,**extra)

    def test_local_raw_state_strict_and_sha(self):
        state=TinyClip().state_dict()
        path=self.local(state); create=unittest.mock.Mock(return_value=TinyClip())
        detector,info=self.build(path,create)
        create.assert_called_once_with('ViT-L-14-quickgelu',pretrained=None,require_pretrained=False)
        for key,value in state.items():
            torch.testing.assert_close(detector.clip_model.state_dict()[key],value,rtol=0,atol=0)
        self.assertEqual(info['initialization']['file_sha256'],mf.file_sha256(path))
        self.assertEqual(info['initialization']['open_clip_version'],'CPU-MOCK-NOT-REAL')
        self.assertEqual(info['initialization']['mode'],'explicit_local')
        self.assertTrue(info['initialization']['strict'])
        self.assertEqual(checkpoint.find_absolute_path_strings(info['initialization']),[])

    def test_head_is_new_not_loaded_from_backbone(self):
        path=self.local()
        a,_=self.build(path); b,_=self.build(path)
        for key,value in a.clip_model.state_dict().items():
            torch.testing.assert_close(value,b.clip_model.state_dict()[key],rtol=0,atol=0)
        self.assertFalse(torch.equal(a.head.weight,b.head.weight))
        torch.testing.assert_close(a.head.bias,torch.zeros(1),rtol=0,atol=0)

    def test_expanded_local_path_hashes_the_same_file(self):
        local=self.local()
        with patch.object(Path,'expanduser',return_value=local):
            _,info=self.build('~/initialization.pt')
        self.assertEqual(info['initialization']['file_sha256'],mf.file_sha256(local))

    def test_local_single_state_wrapper_and_uniform_ddp(self):
        state={'module.'+k:v for k,v in TinyClip().state_dict().items()}
        detector,_=self.build(self.local({'state_dict':state}))
        self.assertEqual(set(detector.clip_model.state_dict()),{k[7:] for k in state})

    def test_real_tiny_torchscript_serialization_cpu_only(self):
        class ScriptClip(TinyClip):
            def forward(self,images): return self.encode_image(images)
        source=ScriptClip().eval()
        path=self.p/'tiny_mock_script.pt'
        torch.jit.save(torch.jit.trace(source,torch.ones(1,3,2,2)),str(path))
        detector,info=self.build(path)
        self.assertEqual(info['initialization']['format'],'openai_torchscript')
        for key,value in source.state_dict().items():
            torch.testing.assert_close(value,detector.clip_model.state_dict()[key],rtol=0,atol=0)
        self.assertLess(path.stat().st_size,100000)

    def test_torchscript_known_metadata_checked(self):
        class ScriptClip(TinyClip):
            def __init__(self):
                super().__init__(); self.register_buffer('input_resolution',torch.tensor(224))
            def forward(self,images): return self.encode_image(images)
        path=self.p/'tiny_script_metadata.pt'
        torch.jit.save(torch.jit.trace(ScriptClip(),torch.ones(1,3,2,2)),str(path))
        _,info=self.build(path)
        self.assertEqual(info['initialization']['removed_torchscript_metadata'],['input_resolution'])
        source=ScriptClip(); source.input_resolution.fill_(336)
        torch.jit.save(torch.jit.trace(source,torch.ones(1,3,2,2)),str(path))
        with self.assertRaisesRegex(ValueError,'metadata'): self.build(path)

    def test_missing_local_file_fails_before_factory(self):
        create=unittest.mock.Mock()
        with self.assertRaises(FileNotFoundError): self.build(self.p/'absent.pt',create)
        create.assert_not_called()

    def test_corrupt_format_fails_before_factory(self):
        path=self.p/'bad.pt'; path.write_bytes(b'not a torch archive')
        create=unittest.mock.Mock()
        with self.assertRaises(ValueError): self.build(path,create)
        create.assert_not_called()

    def test_decd_checkpoint_and_author_wrapper_rejected(self):
        for i,payload in enumerate([{'model_state_dict':TinyClip().state_dict()},
                {'format_version':1,'optimizer_state_dict':{}},
                {'init_origin':'wrapped','model_state_dict':TinyClip().state_dict()},
                {'state_dict':TinyClip().state_dict(),'optimizer':{}}]):
            with self.subTest(i=i),self.assertRaises(ValueError): self.build(self.local(payload,f'bad{i}.pt'))

    def test_missing_unexpected_and_shape_keys_fail_closed(self):
        base=TinyClip().state_dict()
        for kind in ('missing','unexpected','shape','partial_ddp'):
            state=copy.deepcopy(base)
            if kind=='missing': state.pop(next(iter(state)))
            if kind=='unexpected': state['head.weight']=torch.zeros(1,3)
            if kind=='shape': state[next(iter(state))]=torch.zeros(99)
            if kind=='partial_ddp': state['module.'+next(iter(state))]=state.pop(next(iter(state)))
            with self.subTest(kind=kind),self.assertRaises(ValueError): self.build(self.local(state,kind+'.pt'))

    def test_nonfinite_or_nonfloat_initialization_rejected(self):
        for i,value in enumerate([float('nan'),float('inf'),7]):
            state=TinyClip().state_dict(); k=next(iter(state))
            state[k]=(torch.full_like(state[k],value) if i<2 else torch.ones_like(state[k],dtype=torch.int64))
            with self.subTest(i=i),self.assertRaises((ValueError,FloatingPointError)):
                self.build(self.local(state,f'bad_value{i}.pt'))

    def test_official_failure_never_retries_random(self):
        create=unittest.mock.Mock(side_effect=RuntimeError('official load failed'))
        with patch.dict(sys.modules,{'open_clip':types.SimpleNamespace(create_model=create)}):
            with self.assertRaisesRegex(RuntimeError,'official load failed'):
                model.build_detector(model_name=self.cfg['model_name'],pretrained='openai',
                    image_size=224,trainable_blocks=2,device=torch.device('cpu'))
        create.assert_called_once_with('ViT-L-14-quickgelu',pretrained='openai',require_pretrained=True)

    def test_local_factory_failure_never_falls_back(self):
        create=unittest.mock.Mock(side_effect=RuntimeError('construction failed'))
        with self.assertRaises(RuntimeError): self.build(self.local(),create)
        self.assertEqual(create.call_count,1)

    def test_unexpected_strict_loader_result_rejected(self):
        clip=TinyClip()
        clip.load_state_dict=unittest.mock.Mock(return_value=types.SimpleNamespace(missing_keys=['unexpected'],unexpected_keys=[]))
        create=unittest.mock.Mock(return_value=clip)
        with self.assertRaisesRegex(ValueError,'incompatible keys'): self.build(self.local(),create)

    def test_mismatched_architecture_rejected(self):
        with self.assertRaises(ValueError):
            model.build_detector(model_name='ViT-B-32',pretrained='openai',image_size=224,
                trainable_blocks=2,device=torch.device('cpu'),pretrained_path=self.local())

    def test_file_changed_during_load_rejected(self):
        path=self.local()
        with patch.object(model,'_initialization_sha256',side_effect=['a'*64,'b'*64]):
            with self.assertRaisesRegex(ValueError,'changed'): self.build(path)

    def test_init_cannot_enter_checkpoint_restore_mode(self):
        with self.assertRaisesRegex(ValueError,'restoration'):
            self.build(self.local(),load_pretrained=False)

    def test_cli_pretrained_path_parse(self):
        module=runpy.run_path(str(ROOT/'scripts/train_reference.py'))
        with patch.object(sys,'argv',['train_reference.py','--pretrained-path','weights/init.pt']):
            self.assertEqual(module['parse_args']().pretrained_path,Path('weights/init.pt'))

    def test_cli_missing_initialization_fails_before_data(self):
        with self.assertRaises(FileNotFoundError):
            invoke('train_reference.py',['--device','cpu','--pretrained-path',self.p/'absent.pt'])

    def test_reference_local_cli_config_precedence_and_provenance(self):
        local=self.local(); config_path=self.p/'cfg/configs/custom.yaml'
        config_path.parent.mkdir(parents=True)
        cfg=dict(self.cfg,pretrained_path='absent-in-config.pt')
        config_path.write_text(yaml.safe_dump(cfg),encoding='utf-8')
        init_record=dict(mode='explicit_local',file_sha256=mf.file_sha256(local),open_clip_version='CPU-MOCK-NOT-REAL')
        def make(**kw):
            self.assertEqual(kw['pretrained_path'],local)
            detector,info=factory(); info['initialization']=init_record
            return detector,info
        output=self.p/'reference.pth'
        with patch.object(model,'build_detector',side_effect=make):
            invoke('train_reference.py',['--config',config_path,'--pretrained-path',local,
                '--train-manifest',self.paths['train'],'--dev-manifest',self.paths['dev'],
                '--device','cpu','--max-batches',1,'--output',output])
        self.assertEqual(checkpoint.load_checkpoint(output)['metadata']['initialization'],init_record)
        self.assertEqual(json.loads((self.p/'reference_initialization.json').read_text()),init_record)

    def test_config_relative_initialization_and_restore_ignores_stale_path(self):
        config_path=self.p/'cfg/configs/custom.yaml'; config_path.parent.mkdir(parents=True)
        local=self.p/'cfg/weights/init.pt'; local.parent.mkdir()
        torch.save(TinyClip().state_dict(),local)
        cfg=dict(self.cfg,pretrained_path='weights/init.pt')
        config_path.write_text(yaml.safe_dump(cfg),encoding='utf-8')
        seen=[]
        def make(**kw): seen.append(kw['pretrained_path']); return factory()
        with patch.object(model,'build_detector',side_effect=make):
            invoke('train_reference.py',['--config',config_path,'--train-manifest',self.paths['train'],
                '--dev-manifest',self.paths['dev'],'--device','cpu','--max-batches',1,'--output',self.p/'reference.pth'])
        self.assertEqual(seen,[local])
        parent,_,_=self.parent_fixture(); payload=checkpoint.load_checkpoint(parent)
        payload['config']['pretrained_path']='nonexistent-in-another-installation.pt'
        torch.save(payload,parent)
        fake=types.SimpleNamespace(create_model=unittest.mock.Mock(return_value=TinyClip()))
        with patch.dict(sys.modules,{'open_clip':fake}):
            checkpoint.restore_detector(parent,device=torch.device('cpu'),restore_optimizer=True,
                require_reference=True,expected_config=self.cfg,
                train_manifest_sha256=mf.file_sha256(self.paths['train']),
                train_content_sha256=mf.content_sha256(data.load_manifest(self.paths['train'])))
        fake.create_model.assert_called_once_with('ViT-L-14-quickgelu',pretrained=None,require_pretrained=False)

    def test_initialization_sidecar_no_overwrite(self):
        path=self.p/'reference_initialization.json'; path.write_text('preserve')
        with self.assertRaises(FileExistsError):
            invoke('train_reference.py',['--device','cpu','--output',self.p/'reference.pth'])
        self.assertEqual(path.read_text(),'preserve')


class ModelAndMath(Fixture):
    def test_active_root_imports(self):
        for module in [model, data, losses, risk, checkpoint, mf]:
            self.assertEqual(Path(module.__file__).resolve().parent, ROOT/'decd/visual_expert')

    def test_official_pretrained_required(self):
        create = unittest.mock.Mock(return_value=TinyClip())
        with patch.dict(sys.modules, {'open_clip': types.SimpleNamespace(create_model=create)}):
            detector, _ = model.build_detector(model_name=self.cfg['model_name'], pretrained='openai',
                image_size=224, trainable_blocks=2, device=torch.device('cpu'))
        create.assert_called_once_with('ViT-L-14-quickgelu', pretrained='openai', require_pretrained=True)
        self.assertEqual(detector.head.weight.shape, (1, 3))

    def test_restore_construction_never_requests_pretrained(self):
        create = unittest.mock.Mock(return_value=TinyClip())
        with patch.dict(sys.modules, {'open_clip': types.SimpleNamespace(create_model=create)}):
            model.build_detector(model_name=self.cfg['model_name'], pretrained='openai', image_size=224,
                trainable_blocks=2, device=torch.device('cpu'), load_pretrained=False,
                initialize_head=False, feature_dim=3)
        create.assert_called_once_with('ViT-L-14-quickgelu', pretrained=None, require_pretrained=False)

    def test_missing_openclip_fails(self):
        with patch.dict(sys.modules, {'open_clip': None}), self.assertRaises(RuntimeError):
            model.build_detector(model_name=self.cfg['model_name'], pretrained='openai', image_size=224,
                trainable_blocks=2, device=torch.device('cpu'))

    def test_trainable_scope_exact(self):
        detector, info = factory()
        names = info['trainable_parameter_names']
        self.assertEqual(len(names), 8)
        self.assertTrue(all(any(s in n for s in ['resblocks.2.', 'resblocks.3.', 'ln_post.', 'head.']) for n in names))
        self.assertFalse(detector.clip_model.frozen_text_placeholder.requires_grad)

    def test_preprocessing_letterbox(self):
        image = Image.new('RGB', (20, 10), (250, 0, 0))
        tensor = data.LetterboxCLIPTransform()(image)
        self.assertEqual(tuple(tensor.shape), (3, 224, 224))
        gray = (torch.full((3,), 127/255.)-torch.tensor(data.CLIP_MEAN))/torch.tensor(data.CLIP_STD)
        torch.testing.assert_close(tensor[:, 0, 0], gray)

    def test_sampler100_indices_only(self): self.sampler(100)
    def test_sampler50_indices_only(self): self.sampler(50)
    def sampler(self, n):
        rows = [dict(domain=d, label=y) for d, y in EXPECTED_CELLS for _ in range(3)]
        actual = list(DomainLabelBalancedBatchSampler(rows, 6, n, 2026))
        self.assertEqual(len(actual), n)
        self.assertEqual(actual, list(DomainLabelBalancedBatchSampler(rows, 6, n, 2026)))
        for batch in actual:
            self.assertEqual(sorted((rows[i]['domain'], rows[i]['label']) for i in batch), sorted(EXPECTED_CELLS))

    def test_sampler_invalid(self):
        for batch, count in [(0, 1), (7, 1), (6, 0)]:
            with self.subTest(batch=batch, count=count), self.assertRaises(ValueError):
                DomainLabelBalancedBatchSampler([], batch, count, 2026)

    def test_signed_margin_mean_squared(self):
        np.testing.assert_array_equal(risk.signed_margin(np.array([1., 1.]), np.array([0, 1])), [-1., 1.])
        result = risk.summarize_margins(np.array([0., 1.]), np.array([0, 0]), .5)
        self.assertEqual(result['risk_mean_m0p5'], 1.)
        self.assertEqual(result['risk_sq_mean_m0p5'], 1.25)

    def test_minmax_alpha(self):
        np.testing.assert_allclose(risk._normalize_labelwise(np.array([[0., 1.], [1., 3.], [2., 5.]]),
            mode='minmax', alpha=2.), [[0, 0], [.25, .25], [1, 1]])

    def test_tied_risk_preserved(self):
        for value in [0., .5]:
            np.testing.assert_array_equal(risk._normalize_labelwise(np.full((3, 2), value),
                mode='minmax', alpha=1.), np.ones((3, 2)))

    def test_real_only_squared_hinge(self):
        logits = torch.tensor([.2, .3], requires_grad=True)
        value, _ = losses.real_only_margin_repair(logits=logits, labels=torch.tensor([0., 1.]),
            domain_ids=torch.tensor([0, 0]), weight_matrix=torch.ones(3, 2), gamma=.5)
        torch.testing.assert_close(value, torch.tensor(.49))
        torch.testing.assert_close(torch.autograd.grad(value, logits)[0], torch.tensor([1.4, 0.]))

    def test_fake_still_in_detection(self):
        logits = torch.tensor([.2, .3], requires_grad=True)
        gradient = torch.autograd.grad(losses.detection_loss(logits, torch.tensor([0., 1.])), logits)[0]
        self.assertTrue(bool((gradient != 0).all()))

    def test_zero_weights_no_repair(self):
        z = torch.ones(2, requires_grad=True)
        value, _ = losses.real_only_margin_repair(logits=z, labels=torch.tensor([0., 1.]),
            domain_ids=torch.tensor([0, 0]), weight_matrix=torch.zeros(3, 2), gamma=.5)
        self.assertEqual(value.item(), 0.)
        torch.testing.assert_close(torch.autograd.grad(value, z)[0], torch.zeros(2))

    def test_rho_cap_and_zero(self):
        self.assertEqual(losses.adaptive_auxiliary_scale([torch.tensor([3.])], [torch.ones(1)], rho_max=2., epsilon=1e-30)[0], 2.)
        self.assertEqual(losses.adaptive_auxiliary_scale([torch.ones(1)], [torch.zeros(1)], rho_max=10., epsilon=1e-30)[0], 10.)

    def test_ramp(self):
        self.assertEqual([losses.linear_ramp(t, 10) for t in [1, 10, 50]], [.1, 1., 1.])
        self.assertEqual(losses.linear_ramp(1, 0), 1.)

    def test_gradient_combination(self):
        p = torch.nn.Parameter(torch.ones(1))
        losses.assign_combined_gradients(parameters=[p], detection_gradients=[torch.tensor([2.])],
            repair_gradients=[torch.tensor([3.])], repair_lambda=.1, rho=2.)
        torch.testing.assert_close(p.grad, torch.tensor([2.6]))

    def test_nonfinite_failures(self):
        for value in [float('nan'), float('inf'), -float('inf')]:
            with self.subTest(value=value):
                with self.assertRaises(FloatingPointError): losses.detection_loss(torch.tensor([value]), torch.zeros(1))
                with self.assertRaises(FloatingPointError): losses.gradient_l2_norm([torch.tensor([value])])

    def test_prediction_failures(self):
        for mode in ['short', 'nan', 'reverse']:
            fixtures = [dict(image=torch.zeros(2), label=torch.tensor(float(i)), manifest_index=torch.tensor(1-i if mode=='reverse' else i), sample_id=str(i), sha256='', domain='Chart') for i in range(2)]
            class Detector(torch.nn.Module):
                def forward(self, x):
                    return torch.full((len(x)-int(mode=='short'),), float('nan') if mode=='nan' else 0.), x
            with self.subTest(mode=mode), self.assertRaises((ValueError, FloatingPointError)):
                model.collect_predictions(Detector(), DataLoader(fixtures, batch_size=2), device=torch.device('cpu'), amp_enabled=False)

    def test_ap_uses_continuous_scores(self):
        actual = model.binary_metrics(np.array([1, 0, 1, 0]), np.array([.9, .8, .7, .1]), .5)
        self.assertAlmostEqual(actual['ap'], 5/6)
        self.assertEqual(actual['accuracy'], .75)


class ManifestChecks(Fixture):
    def raw(self):
        return [dict(file_name=r['image_path'], domain='Photo' if r['domain']=='Photo_and_poster' else r['domain'],
                     label=f"{r['label']}_"+('real' if r['label']==0 else 'fake'), sha256=r['sha256']) for r in self.rows['train']]

    def test_public_metadata(self):
        rows = mf.read_manifest(csv_write(self.p/'metadata.csv', self.raw()), image_root=self.p, require_sha=True)
        self.assertEqual(len(rows), 6); self.assertEqual({r['domain'] for r in rows}, set(mf.DOMAIN_TO_ID))
        self.assertEqual({r['label'] for r in rows}, {0, 1})

    def test_missing_and_bad_sha(self):
        for i, change in enumerate([dict(file_name='absent.png'), dict(sha256='0'*64)]):
            row = dict(self.raw()[0], **change)
            with self.subTest(change=change), self.assertRaises((ValueError, FileNotFoundError)):
                mf.read_manifest(csv_write(self.p/f'bad{i}.csv', [row]), image_root=self.p, require_sha=True)

    def test_path_escape(self):
        with self.assertRaises(ValueError):
            mf.read_manifest(csv_write(self.p/'escape.csv', [dict(self.raw()[0], file_name='../outside.png')]), image_root=self.p)

    def test_invalid_labels(self):
        for label in ['2', 'NaN', '']:
            with self.subTest(label=label), self.assertRaises(ValueError): mf.parse_label({'label': label})
        with self.assertRaises(ValueError): mf.parse_label({'label': '0', 'label_name': 'Fake'})

    def test_duplicates(self):
        row = self.raw()[0]
        with self.assertRaises(ValueError): mf.read_manifest(csv_write(self.p/'dups.csv', [row, row]), image_root=self.p)

    def test_original_current_cross_identity(self):
        a, b = [dict(r) for r in self.rows['train'][:2]]
        a['original_sha256'] = b['sha256']
        with self.assertRaises(ValueError): data.load_manifest(csv_write(self.p/'cross.csv', [a, b]))

    def test_source_vs_standardized_hash(self):
        row = self.raw()[0]; output = self.p/'derived.jpg'
        with Image.open(self.p/row['file_name']) as image: image.convert('RGB').save(output)
        row.update(file_name=output.name, standardized_sha256=mf.file_sha256(output))
        actual = mf.read_manifest(csv_write(self.p/'legacy.csv', [row]), image_root=self.p)[0]
        self.assertEqual(actual['original_sha256'], self.raw()[0]['sha256'])
        self.assertEqual(actual['sha256'], mf.file_sha256(output))

    def test_corrupt_image_not_zero(self):
        image = self.p/'corrupt.png'; image.write_bytes(b'not an image')
        row = dict(self.raw()[0], file_name=image.name, sha256=mf.file_sha256(image))
        rows = mf.read_manifest(csv_write(self.p/'corrupt.csv', [row]), image_root=self.p)
        with self.assertRaises(RuntimeError): data.ManifestDataset(rows)[0]

    def test_roles_forbid_benchmark(self):
        for role in ['train', 'dev']:
            with self.subTest(role=role), self.assertRaises(ValueError):
                mf.assert_role([dict(split=' test ', benchmark_id='mosaic_benchmark_6000_v1')], role)

    def test_unassigned_selected_pool_cannot_enter_train_or_dev(self):
        for role in ('train','dev'):
            with self.subTest(role=role),self.assertRaises(ValueError):
                mf.assert_role([dict(split='unassigned',benchmark_id='')],role)

    def test_train_dev_identity_group(self):
        a = data.load_manifest(self.paths['train']); b = data.load_manifest(self.paths['dev'])
        mf.assert_disjoint(a, b)
        with self.assertRaises(ValueError): mf.assert_disjoint(a, a)
        b[0]['group_id'] = a[0]['group_id']
        with self.assertRaises(ValueError): mf.assert_disjoint(a, b)

    def test_explicit_converter(self):
        metadata = csv_write(self.p/'pool.csv', self.raw())
        selection = csv_write(self.p/'selected.csv', [{'file_name': r['file_name']} for r in self.raw()])
        output = self.p/'converted/train.csv'
        result = CONVERT['convert'](metadata, self.p, selection, output, 'train')
        self.assertEqual(result['rows'], 6); self.assertFalse(result['automatic_split'])
        self.assertEqual(len(data.load_manifest(output, role='train')), 6)
        with self.assertRaises(FileExistsError): CONVERT['convert'](metadata, self.p, selection, output, 'train')

    def test_converter_does_not_invent_selection(self):
        metadata = csv_write(self.p/'pool.csv', self.raw())
        missing = self.p/'missing.csv'
        with self.assertRaises(FileNotFoundError): CONVERT['convert'](metadata, self.p, missing, self.p/'out.csv', 'train')

    def test_selection_group_and_id_preserved(self):
        metadata=csv_write(self.p/'pool.csv',self.raw())
        selection=csv_write(self.p/'selection.csv',[dict(file_name=self.raw()[0]['file_name'],
            group_id='known-source-family',sample_id='explicit-stable-id')])
        output=self.p/'selected.csv'; CONVERT['convert'](metadata,self.p,selection,output,'train')
        row=data.load_manifest(output,role='train')[0]
        self.assertEqual(row['group_id'],'known-source-family')
        self.assertEqual(row['sample_id'],'explicit-stable-id')

    def test_selection_label_and_sha_conflicts_rejected(self):
        metadata=csv_write(self.p/'pool.csv',self.raw())
        for i,extra in enumerate([dict(label=1),dict(sha256='0'*64)]):
            selection=csv_write(self.p/f'selection{i}.csv',[dict(file_name=self.raw()[0]['file_name'],**extra)])
            with self.subTest(extra=extra),self.assertRaises(ValueError):
                CONVERT['convert'](metadata,self.p,selection,self.p/f'out{i}.csv','train')

    def test_explicit_bad_field(self):
        with self.assertRaises(ValueError): data.load_manifest(self.paths['train'], require_domain=False, domain_field='absent')


class CheckpointAndProfile(Fixture):
    def restore(self, path):
        with patch.object(checkpoint, 'build_detector', factory):
            return checkpoint.restore_detector(path, device=torch.device('cpu'), restore_optimizer=True,
                require_reference=True, expected_config=self.cfg,
                train_manifest_sha256=mf.file_sha256(self.paths['train']),
                train_content_sha256=mf.content_sha256(data.load_manifest(self.paths['train'])))

    def test_roundtrip_model_and_adamw(self):
        path, original, original_opt = self.parent_fixture()
        restored, optimizer, _, _ = self.restore(path)
        torch.testing.assert_close(original(torch.ones(2, 3, 2, 2))[0], restored(torch.ones(2, 3, 2, 2))[0], rtol=0, atol=0)
        for a, b in zip(original_opt.state.values(), optimizer.state.values()):
            for key in a: torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)

    def test_reject_wrong_parent_sources(self):
        path, _, _ = self.parent_fixture(); baseline = checkpoint.load_checkpoint(path)
        for i, kind in enumerate(['stage', 'steps', 'manifest', 'config', 'optimizer_step', 'missing_key', 'moment_shape', 'nan']):
            payload = copy.deepcopy(baseline)
            if kind=='stage': payload['metadata']['stage']='visual_expert'
            if kind=='steps': payload['metadata']['completed_batches']=2
            if kind=='manifest': payload['metadata']['train_manifest_sha256']='0'*64
            if kind=='config': payload['config']['seed']=2027
            if kind=='optimizer_step': next(iter(payload['optimizer_state_dict']['state'].values()))['step'].fill_(2)
            if kind=='missing_key': payload['model_state_dict'].pop('head.bias')
            if kind=='moment_shape': next(iter(payload['optimizer_state_dict']['state'].values()))['exp_avg']=torch.zeros(17)
            if kind=='nan': payload['model_state_dict']['head.bias'].fill_(float('nan'))
            output=self.p/f'bad{i}.pth'; torch.save(payload, output)
            with self.subTest(kind=kind), self.assertRaises((ValueError, RuntimeError, FloatingPointError)): self.restore(output)

    def test_missing_checkpoint(self):
        with self.assertRaises(FileNotFoundError): checkpoint.load_checkpoint(self.p/'absent.pth')

    def test_save_nonfinite_rejected(self):
        detector, _ = factory(); detector.head.bias.data.fill_(float('nan'))
        with self.assertRaises(FloatingPointError): checkpoint.save_checkpoint(self.p/'bad.pth', detector=detector,
            optimizer=None, config=self.cfg, feature_dim=3)
        self.assertFalse((self.p/'bad.pth').exists())

    def test_config_defaults_and_invalid(self):
        checkpoint.validate_training_config(self.cfg)
        self.assertEqual((self.cfg['reference_batches'], self.cfg['repair_batches'], self.cfg['risk_column']), (100, 50, 'risk_mean_m0p5'))
        for field, value in [('repair_label_side','fake'), ('rho_max',float('nan')), ('reference_batches',99), ('batch_size',7)]:
            with self.subTest(field=field), self.assertRaises(ValueError): checkpoint.validate_training_config(dict(self.cfg, **{field:value}))

    def test_profile_consistency_and_provenance(self):
        preds=[dict(domain=d, label=y, logit=float(i)) for i,(d,y) in enumerate(EXPECTED_CELLS)]
        profile=risk.build_risk_profile(preds, gamma=.5, risk_column='risk_mean_m0p5', normalization='minmax', alpha=1.,
            provenance=dict(manifest_sha256='a'*64, reference_checkpoint_sha256='b'*64))
        kw=dict(risk_column='risk_mean_m0p5', gamma=.5, normalization='minmax', normalization_alpha=1.,
                train_manifest_sha256='a'*64, reference_checkpoint_sha256='b'*64)
        risk.validate_risk_profile(profile, **kw)
        for kind in ['weight', 'cell', 'source']:
            bad=copy.deepcopy(profile)
            if kind=='weight': bad['weight_matrix'][0][0]=.123
            if kind=='cell': bad['cells'][0]['n']=0
            if kind=='source': bad['provenance']['manifest_sha256']='c'*64
            with self.subTest(kind=kind), self.assertRaises(ValueError): risk.validate_risk_profile(bad, **kw)

    def test_existing_logs_are_protected(self):
        for script, filename in [('train_reference.py','reference_metrics.json'), ('train_visual_expert.py','visual_expert_train_log.json')]:
            folder=self.p/script; folder.mkdir(); (folder/filename).write_text('preserve', encoding='utf-8')
            with self.subTest(script=script), self.assertRaises(FileExistsError):
                invoke(script, ['--device','cpu','--output',folder/'new.pth'])
            self.assertEqual((folder/filename).read_text(), 'preserve')


class BenchmarkChecks(Fixture):
    def test_balanced6000_deterministic(self):
        rows=virtual_pool(); selected, rest, _, _=BENCH['select_stratified'](rows)
        again, _, _, _=BENCH['select_stratified'](list(reversed(rows)))
        self.assertEqual([r['sha256'] for r in selected], [r['sha256'] for r in again])
        self.assertEqual((len(selected), len(rest)), (6000,18))
        self.assertEqual(len({r['sha256'] for r in selected}),6000)
        for d,y in EXPECTED_CELLS: self.assertEqual(sum(r['domain']==d and r['label']==y for r in selected),1000)

    def test_exclusion_deduplication(self):
        rows=virtual_pool(); h=rows[0]['sha256']; rows[1]['group_id']='heldout'
        selected, _, rejected, _=BENCH['select_stratified'](rows+[dict(rows[2])], exclusion_hashes={h}, exclusion_groups={'heldout'})
        self.assertEqual(len(rejected),3)
        self.assertFalse(any(r['sha256']==h or r['group_id']=='heldout' for r in selected))

    def test_transitive_original_exclusion(self):
        rows=virtual_pool(); a,b=rows[:2]
        alias=dict(b, sample_id='converted-alias', original_sha256=a['sha256'])
        selected, _, rejected, _=BENCH['select_stratified'](rows+[alias], exclusion_hashes={a['sha256']})
        self.assertEqual(len(rejected),3)
        self.assertFalse({a['sha256'],b['sha256']} & mf.identities(selected))

    def test_group_exclusion_propagates_through_alias(self):
        rows=virtual_pool(); a,b=rows[:2]; a['group_id']=b['group_id']='family'
        alias=dict(a, sample_id='alias', group_id='')
        selected, _, rejected, _=BENCH['select_stratified'](rows+[alias], exclusion_groups={'family'})
        self.assertEqual(len(rejected),3)
        self.assertFalse({a['sha256'],b['sha256']} & mf.identities(selected))

    def test_insufficient_no_replacement(self):
        rows=virtual_pool(999)
        with self.assertRaises(ValueError): BENCH['select_stratified'](rows)

    def test_exclusion_can_make_insufficient(self):
        rows=virtual_pool(1000)
        with self.assertRaises(ValueError): BENCH['select_stratified'](rows,exclusion_hashes={rows[0]['sha256']})

    def test_conflicting_duplicate_or_id(self):
        rows=virtual_pool()
        with self.assertRaises(ValueError): BENCH['select_stratified'](rows+[dict(rows[0],label=1)])
        with self.assertRaises(ValueError): BENCH['select_stratified'](rows+[dict(rows[1],sample_id=rows[0]['sample_id'])])

    def test_assigned_pool_rejected(self):
        with self.assertRaises(ValueError): BENCH['freeze_benchmark'](self.paths['train'], self.p, self.p/'freeze')
        self.assertFalse((self.p/'freeze').exists())

    def test_real_synthetic_freeze_verify_immutable(self):
        # 6012 distinct1x1 PNGs, ~0.4MB payload; no real MOSAIC data.
        images=self.p/'images'; images.mkdir(); rows=[]; i=0
        for d in mf.DOMAIN_TO_ID:
            for y in (0,1):
                for _ in range(1002):
                    path=images/f'{i}.png'; Image.new('RGB',(1,1),(i%256,(i//256)%256,15)).save(path)
                    rows.append(dict(file_name=path.name,domain='Photo' if d=='Photo_and_poster' else d,
                        label=f'{y}_'+('real' if y==0 else 'fake'),sha256=mf.file_sha256(path),group_id='heldout' if i<2 else ''))
                    i+=1
        metadata=csv_write(self.p/'metadata.csv', rows)
        exclusion=csv_write(self.p/'exclusion.csv',[dict(sha256=rows[0]['sha256'],group_id='heldout')])
        before={p.name:mf.file_sha256(p) for p in images.iterdir()}
        output=self.p/'frozen'
        report=BENCH['freeze_benchmark'](metadata,images,output,exclusions=[exclusion])
        self.assertEqual((report['selected_rows'],report['remaining_rows'],report['rejected_rows']),(6000,10,2))
        self.assertEqual(report['required_partition_counts'],dict(train=4200,dev=600,test=1200))
        self.assertEqual(report['required_per_domain_label'],dict(train=700,dev=100,test=200))
        self.assertEqual(report['formal_split_status'],'NOT_CREATED_REUSE_OR_AUTHOR_APPROVAL_REQUIRED')
        self.assertEqual(report['isolation']['near_duplicate'],'NOT_VERIFIED')
        self.assertEqual(report['isolation']['source_group_isolation'],'NOT_VERIFIED')
        frozen_before={p.name:mf.file_sha256(p) for p in output.iterdir()}
        BENCH['verify_frozen'](output,metadata=metadata,image_root=images,exclusions=[exclusion])
        self.assertEqual(frozen_before,{p.name:mf.file_sha256(p) for p in output.iterdir()})
        with self.assertRaises(FileExistsError): BENCH['freeze_benchmark'](metadata,images,output,seed=2027)
        self.assertEqual(before,{p.name:mf.file_sha256(p) for p in images.iterdir()})
        self.assertTrue(all(r['split']=='unassigned' for r in mf.read_csv(output/'mosaic_remaining.csv')[1]))
        selected_rows=mf.read_csv(output/'mosaic_selected_6000.csv')[1]
        self.assertTrue(all(r['split']=='unassigned' and not r['benchmark_id'] for r in selected_rows))
        self.assertFalse((output/'mosaic_benchmark_6000.csv').exists())
        self.assertTrue(all(not (output/name).exists() for name in ['train.csv','dev.csv','test.csv']))
        with self.assertRaises(ValueError): data.load_manifest(output/'mosaic_selected_6000.csv',role='train')
        with (output/'mosaic_selected_6000.csv').open('a',encoding='utf-8') as stream: stream.write('tampered\n')
        with self.assertRaises(ValueError): BENCH['verify_frozen'](output,exclusions=[exclusion])

    def test_legacy_test_only_mock_freeze_verified_without_migration(self):
        # Mock legacy ledger only: no real images and no approved final split.
        rows=virtual_pool(1000)
        for r in rows: r.update(split='test',benchmark_id=BENCH['BENCHMARK_ID'])
        output=self.p/'legacy'; output.mkdir()
        for name in ['mosaic_benchmark_6000.csv','mosaic_remaining.csv','mosaic_rejected.csv']:
            (output/name).write_text('MOCK_LEGACY_ONLY\n',encoding='utf-8')
        record=dict(protocol=BENCH['BENCHMARK_ID'],selected_rows=6000,exclusions=[],
            outputs={p.name:mf.file_sha256(p) for p in output.iterdir()},
            selected_identities=[dict(sample_id=r['sample_id'],original_sha256=r['original_sha256'],sha256=r['sha256']) for r in rows])
        freeze=output/'benchmark_freeze.json'; freeze.write_text(json.dumps(record),encoding='utf-8')
        (output/'benchmark_freeze.sha256').write_text(mf.file_sha256(freeze),encoding='ascii')
        before={p.name:mf.file_sha256(p) for p in output.iterdir()}
        read=unittest.mock.Mock(return_value=rows)
        with patch.dict(BENCH['verify_frozen'].__globals__,{'read_manifest':read}):
            result=BENCH['verify_frozen'](output)
        self.assertTrue(result['legacy_test_only_artifact'])
        self.assertEqual(result['final_mosaic6000_split_status'],'NOT_VERIFIED_NOT_A_FORMAL_DATASET_SPLIT')
        self.assertEqual(before,{p.name:mf.file_sha256(p) for p in output.iterdir()})
        self.assertFalse((output/'selection_freeze.json').exists())


class BoundedIntegration(Fixture):
    def test_short_reference_rejected_as_full_parent(self):
        output=self.p/'reference_debug.pth'
        with patch.object(model,'build_detector',factory):
            invoke('train_reference.py',['--train-manifest',self.paths['train'],'--dev-manifest',self.paths['dev'],
                '--device','cpu','--max-batches',2,'--output',output])
        payload=checkpoint.load_checkpoint(output)
        self.assertEqual(payload['metadata']['completed_batches'],2)
        self.assertTrue(all(s['step'].item()==2 for s in payload['optimizer_state_dict']['state'].values()))
        with self.assertRaises(ValueError):
            invoke('build_risk_profile.py',['--manifest',self.paths['train'],'--checkpoint',output,
                '--output',self.p/'must_not_profile.json','--device','cpu'])
        self.assertFalse((self.p/'must_not_profile.json').exists())

    def test_mock_endpoint_profile_one_repair_and_evaluation(self):
        parent,_,_=self.parent_fixture(); profile=self.p/'risk.json'; final=self.p/'debug_repair.pth'
        with patch.object(checkpoint,'build_detector',factory):
            invoke('build_risk_profile.py',['--manifest',self.paths['train'],'--checkpoint',parent,'--output',profile,'--device','cpu'])
            digest=mf.file_sha256(profile)
            invoke('train_visual_expert.py',['--train-manifest',self.paths['train'],'--dev-manifest',self.paths['dev'],
                '--reference-checkpoint',parent,'--risk-profile',profile,'--output',final,'--device','cpu','--max-batches',1])
            invoke('evaluate_visual_expert.py',['--manifest',self.paths['dev'],'--checkpoint',final,
                '--output-dir',self.p/'evaluation','--device','cpu'])
        payload=checkpoint.load_checkpoint(final)
        self.assertEqual(payload['metadata']['completed_batches'],1)
        self.assertTrue(all(s['step'].item()==101 for s in payload['optimizer_state_dict']['state'].values()))
        self.assertEqual(digest,mf.file_sha256(profile))
        self.assertEqual(len(mf.read_csv(self.p/'evaluation/predictions.csv')[1]),6)
        self.assertLess(final.stat().st_size,100000)

    def test_data_smoke_subprocess_no_backbone(self):
        process=subprocess.run([sys.executable,'-B',str(ROOT/'tools/visual_cpu_smoke.py'),
            '--output-dir',str(self.p/'smoke')],cwd=ROOT,env=dict(os.environ),capture_output=True,text=True,timeout=60)
        self.assertEqual(process.returncode,0,process.stderr)
        report=json.loads((self.p/'smoke/smoke_report.json').read_text())
        self.assertFalse(report['model_loaded']); self.assertEqual(report['optimizer_updates'],0)
        self.assertFalse(list((self.p/'smoke').glob('*.pth')))

    def test_cli_batch_bounds_and_evaluation_overwrite(self):
        for name, maximum in [('train_reference.py',100),('train_visual_expert.py',50)]:
            for value in [0,-1,maximum+1]:
                with self.subTest(name=name,value=value), self.assertRaises(ValueError):
                    invoke(name,['--device','cpu','--max-batches',value])
        output=self.p/'evaluation'; output.mkdir(); (output/'metrics.json').write_text('{}')
        with self.assertRaises(FileExistsError): invoke('evaluate_visual_expert.py',['--device','cpu','--output-dir',output])


class ReleasePolicy(unittest.TestCase):
    def test_mosaic6000_is_full_dataset_not_test_only(self):
        text=(ROOT/'README.md').read_text(encoding='utf-8').replace(',','')
        for row in ['| Train | 700 | 4200 |','| Dev | 100 | 600 |','| Radar/Test | 200 | 1200 |']:
            self.assertIn(row,text)
        self.assertIn('Historical Train6189/Dev597',text)
        self.assertNotIn('new test-only candidate benchmark',text)
    def test_public_config_has_no_server_initialization_path(self):
        cfg=yaml.safe_load((ROOT/'configs/visual_expert_reported.yaml').read_text())
        self.assertIsNone(cfg['pretrained_path']); self.assertEqual(cfg['pretrained'],'openai')
        for path in (ROOT/'configs').glob('visual_expert_reported.yaml'):
            self.assertNotIn('/'+ '/'.join(['root','autodl-tmp']),path.read_text())

    def test_allowlist_exists_code_only(self):
        files=(ROOT/'decd/visual_expert/RELEASE_FILES.txt').read_text().splitlines()
        self.assertEqual(len(files),len(set(files)))
        for name in files:
            with self.subTest(name=name):
                path=(ROOT/name).resolve(); self.assertTrue(path.is_relative_to(ROOT)); self.assertTrue(path.is_file())
                self.assertNotIn(path.suffix.lower(),{'.pth','.pt','.ckpt','.safetensors','.bin','.csv','.png'})

    def test_readme_code_only_and_shared_dependencies(self):
        text=(ROOT/'README.md').read_text(encoding='utf-8')
        self.assertIn('This repository provides training code only. Trained model checkpoints are not distributed.',text)
        self.assertIn('requirements.txt',text)
        self.assertIn('Visual Expert',text)
        req=(ROOT/'requirements.txt').read_text(encoding='utf-8')
        self.assertIn('open-clip-torch==3.3.0',req)
        self.assertIn('transformers>=4.49,<5',req)
        active='\n'.join(line for line in req.splitlines() if not line.lstrip().startswith('#'))
        self.assertNotIn('pytest>=',active.lower())

    def test_single_readme_and_requirements(self):
        files=(ROOT/'decd/visual_expert/RELEASE_FILES.txt').read_text().splitlines()
        for name in ['README.md','requirements.txt']:
            self.assertTrue((ROOT/name).is_file()); self.assertIn(name,files)
        for name in ['README_VISUAL_EXPERT.md','requirements-visual.txt']:
            self.assertFalse((ROOT/name).exists()); self.assertNotIn(name,files)
            self.assertNotIn(name,(ROOT/'README.md').read_text(encoding='utf-8'))

    def test_cuda_not_initialized(self): self.assertFalse(torch.cuda.is_initialized())


class DistributionChecks(Fixture):
    def test_collect_current_whitelist_only(self):
        blobs=PACKAGE['collect'](ROOT)
        expected=(ROOT/'decd/visual_expert/RELEASE_FILES.txt').read_text().splitlines()
        self.assertEqual(set(blobs),set(expected))
        self.assertIn('README.md',blobs)
        self.assertIn('requirements.txt',blobs)
        self.assertNotIn('README_VISUAL_EXPERT.md',blobs)
        self.assertNotIn('requirements-visual.txt',blobs)
        self.assertIn('tools/package_visual_release.py',blobs)

    def test_build_deterministic_safe_zip_no_overwrite(self):
        import zipfile
        with tempfile.TemporaryDirectory(prefix='visual-rc1-package-',dir=self.p) as temp:
            p=Path(temp); a=PACKAGE['build'](ROOT,p/'first'); b=PACKAGE['build'](ROOT,p/'second')
            self.assertEqual(a['sha256'],b['sha256'])
            self.assertEqual(a['files'],len(PACKAGE['collect'](ROOT)))
            with zipfile.ZipFile(p/'first'/a['file']) as zipped:
                self.assertIsNone(zipped.testzip())
                for name,digest in a['members_sha256'].items():
                    self.assertEqual(hashlib.sha256(zipped.read(name)).hexdigest(),digest)
            with self.assertRaises(FileExistsError): PACKAGE['build'](ROOT,p/'first')

    def test_packager_rejects_escape_weights_and_sensitive_text(self):
        with tempfile.TemporaryDirectory(prefix='visual-rc1-policy-',dir=self.p) as temp:
            p=Path(temp); allow=p/'decd/visual_expert/RELEASE_FILES.txt'; allow.parent.mkdir(parents=True)
            for name in ('../outside.py','weights.pt','outputs/result.py'):
                allow.write_text(name+'\n')
                with self.subTest(name=name),self.assertRaises(ValueError): PACKAGE['collect'](p)
            allow.write_text('code.py\n')
            (p/'code.py').write_text("path = '"+'/'+ '/'.join(['root','autodl-tmp','private'])+"'\n")
            with self.assertRaises(ValueError): PACKAGE['collect'](p)


if __name__=='__main__': unittest.main(verbosity=2)
