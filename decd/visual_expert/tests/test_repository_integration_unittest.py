"""CPU-only root-repository checks; synthetic artifacts, no pretrained models.

Run from repository root:
  python -B -m unittest discover -s decd/visual_expert/tests \
      -p test_repository_integration_unittest.py -v

Expected failures are reproducible NON-Visual issues intentionally not fixed.
They are not successful product acceptance tests.
"""
from __future__ import annotations

import copy
import csv
import importlib.util
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

import numpy as np
import torch
import yaml
from PIL import Image
torch.set_num_threads(1)

from decd.visual_expert import checkpoint, data, losses, model, risk
from decd.visual_expert.manifests import (
    DOMAIN_TO_ID, assert_disjoint, content_sha256, file_sha256, read_manifest,
)
from decd.visual_expert.sampler import DomainLabelBalancedBatchSampler, EXPECTED_CELLS


def write_csv(path, rows):
    with Path(path).open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    return Path(path)


class TinyVisual(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = torch.nn.Module()
        self.transformer.resblocks = torch.nn.ModuleList([torch.nn.Linear(3, 3) for _ in range(4)])
        self.ln_post = torch.nn.LayerNorm(3)


class TinyClip(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.visual = TinyVisual()
        self.unused_frozen = torch.nn.Parameter(torch.ones(1))

    def encode_image(self, x, normalize=False):
        x = x.mean((2, 3))
        for block in self.visual.transformer.resblocks: x = torch.tanh(block(x))
        return self.visual.ln_post(x)


def tiny_detector(**kwargs):
    detector = model.CLIPBinaryDetector(TinyClip(), 3)
    info = model.configure_trainable_parameters(detector, 2); info['feature_dim'] = 3
    return detector, info


class TinyFrozenBackbone:
    """Deterministic synthetic feature interface, never a Qwen reproduction."""
    def __init__(self, *args, **kwargs): self.feature_dim = 8

    def encode(self, images, prompts):
        rows = []
        for image in images:
            r, g, b = np.asarray(image, dtype=np.float32).mean((0, 1)) / 255.
            base = torch.tensor([r, g, b, r*g, g*b, r*b, (r+g+b)/3, 1.])
            rows.append(torch.stack([base + k * .01 for k in range(len(prompts))]))
        return torch.stack(rows).float()


class Fixtures(unittest.TestCase):
    def setUp(self):
        base = Path(os.environ.get('DECD_TEST_TMP', tempfile.gettempdir())).resolve()
        base.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(prefix='decd-root-cpu-', dir=base)
        self.p = Path(self.tmp.name).resolve()
        assert self.p.is_relative_to(base)  # cleanup is confined to new synthetic artifacts
        self.addCleanup(self.tmp.cleanup)
        self.config = yaml.safe_load((ROOT/'configs/visual_expert_reported.yaml').read_text())
        self.rows = {}; self.manifests = {}
        for cohort, split in enumerate(['train', 'dev', 'test']):
            rows = []
            for d, domain in enumerate(DOMAIN_TO_ID):
                for label in (0, 1):
                    image = self.p/f'{split}_{d}_{label}.png'
                    Image.new('RGB', (9, 7), (d*70, label*160, 25+cohort*65)).save(image)
                    sid = f'{split}_{d}_{label}'
                    rows.append(dict(sample_id=sid, image_path=image.name, domain=domain,
                                     label=label, sha256=file_sha256(image), split=split,
                                     group_id=f'group_{sid}'))
            self.rows[split] = rows
            self.manifests[split] = write_csv(self.p/f'{split}.csv', rows)

    def run_cli(self, script, args):
        with patch.object(sys, 'argv', [script] + list(map(str, args))):
            return runpy.run_path(str(ROOT/'scripts'/script), run_name='__main__')

    def fake_parent(self):
        det, info = tiny_detector()
        opt = model.build_optimizer(info, backbone_lr=1e-5, head_lr=1e-4, weight_decay=1e-4)
        det(torch.ones(2, 3, 2, 2))[0].sum().backward(); opt.step()
        for state in opt.state.values(): state['step'].fill_(100)
        rows = data.load_manifest(self.manifests['train'], role='train')
        target = self.p/'parent.pth'
        checkpoint.save_checkpoint(target, detector=det, optimizer=opt, config=self.config, feature_dim=3,
            metadata=dict(stage='reference', completed_batches=100,
                          train_manifest_sha256=file_sha256(self.manifests['train']),
                          train_content_sha256=content_sha256(rows)))
        return target, rows


class VisualChecks(Fixtures):
    def test_01_active_imports_are_root_modules(self):
        for module in [checkpoint, data, losses, model, risk]:
            self.assertEqual(Path(module.__file__).resolve().parent, ROOT/'decd/visual_expert')

    def test_02_default_algorithm_configuration_unchanged(self):
        self.assertEqual(self.config['risk_column'], 'risk_mean_m0p5')
        self.assertEqual((self.config['reference_batches'], self.config['repair_batches']), (100, 50))
        self.assertEqual((self.config['gamma'], self.config['repair_lambda'], self.config['rho_max']), (.5, .1, 10.))
        checkpoint.validate_training_config(self.config)
        alternate = dict(self.config, gamma=.25, risk_column='risk_mean_m0p25')
        checkpoint.validate_training_config(alternate)  # do not remove existing configurable variants

    def test_03_sampler_100(self): self.check_sampler(100)
    def test_04_sampler_50(self): self.check_sampler(50)

    def check_sampler(self, n):
        rows = [dict(domain=d, label=y) for d, y in EXPECTED_CELLS for _ in range(3)]
        a = list(DomainLabelBalancedBatchSampler(rows, 6, n, 2026))
        self.assertEqual(len(a), n)
        self.assertEqual(a, list(DomainLabelBalancedBatchSampler(rows, 6, n, 2026)))
        for batch in a:
            self.assertEqual(sorted((rows[i]['domain'], rows[i]['label']) for i in batch), sorted(EXPECTED_CELLS))

    def test_05_mean_and_squared_hinge_are_distinct(self):
        values = risk.summarize_margins(np.array([0., 1.]), np.array([0, 0]), .5)
        self.assertEqual(values['risk_mean_m0p5'], 1.)
        self.assertEqual(values['risk_sq_mean_m0p5'], 1.25)

    def test_06_tied_risk_stays_all_one_not_paper_zero(self):
        np.testing.assert_array_equal(risk._normalize_labelwise(np.full((3, 2), .5), mode='minmax', alpha=1.), np.ones((3, 2)))

    def test_07_real_only_repair_gradients(self):
        z = torch.tensor([.2, .3], requires_grad=True)
        loss, _ = losses.real_only_margin_repair(logits=z, labels=torch.tensor([0., 1.]),
                      domain_ids=torch.tensor([0, 0]), weight_matrix=torch.ones(3, 2), gamma=.5)
        grad = torch.autograd.grad(loss, z)[0]
        torch.testing.assert_close(grad, torch.tensor([1.4, 0.]))

    def test_08_rho_ramp_combined_gradient(self):
        rho, _ = losses.adaptive_auxiliary_scale([torch.tensor([3.])], [torch.tensor([1.])], rho_max=2., epsilon=1e-30)
        self.assertEqual(rho, 2.)
        self.assertEqual([losses.linear_ramp(t, 10) for t in [1, 10, 50]], [.1, 1., 1.])
        p = torch.nn.Parameter(torch.ones(1))
        losses.assign_combined_gradients(parameters=[p], detection_gradients=[torch.tensor([2.])],
            repair_gradients=[torch.tensor([3.])], repair_lambda=.1, rho=2.)
        torch.testing.assert_close(p.grad, torch.tensor([2.6]))

    def test_09_nonfinite_loss_and_gradient_rejected(self):
        for value in [float('nan'), float('inf'), -float('inf')]:
            with self.subTest(value=value):
                with self.assertRaises(FloatingPointError): losses.detection_loss(torch.tensor([value]), torch.tensor([0.]))
                with self.assertRaises(FloatingPointError): losses.gradient_l2_norm([torch.tensor([value])])

    def test_10_hf_metadata_schema(self):
        raw = [dict(file_name=r['image_path'], domain='Photo' if r['domain']=='Photo_and_poster' else r['domain'],
                    label='0_real' if r['label']==0 else '1_fake', sha256=r['sha256']) for r in self.rows['train']]
        rows = read_manifest(write_csv(self.p/'hf.csv', raw), image_root=self.p, require_sha=True)
        self.assertEqual({r['domain'] for r in rows}, set(DOMAIN_TO_ID))
        self.assertEqual({r['label'] for r in rows}, {0, 1})

    def test_11_wrong_sha_rejected(self):
        rows = [dict(self.rows['train'][0], sha256='0'*64)]
        with self.assertRaises(ValueError): data.load_manifest(write_csv(self.p/'bad.csv', rows))

    def test_12_missing_and_corrupt_images_rejected(self):
        raw = dict(self.rows['train'][0], image_path='absent.png')
        with self.assertRaises(FileNotFoundError): data.load_manifest(write_csv(self.p/'missing.csv', [raw]))
        bad = self.p/'empty.png'; bad.write_bytes(b'')
        raw.update(image_path=bad.name, sha256=file_sha256(bad))
        rows = data.load_manifest(write_csv(self.p/'corrupt.csv', [raw]))
        with self.assertRaises(RuntimeError): data.ManifestDataset(rows)[0]

    def test_13_duplicate_identity_rejected(self):
        raw = self.rows['train'][0]
        with self.assertRaises(ValueError): data.load_manifest(write_csv(self.p/'duplicate.csv', [raw, raw]))

    def test_14_path_escape_rejected(self):
        raw = dict(self.rows['train'][0], image_path='../escape.png')
        with self.assertRaises(ValueError): read_manifest(write_csv(self.p/'escape.csv', [raw]), image_root=self.p)

    def test_15_train_dev_identity_and_group_isolation(self):
        train = data.load_manifest(self.manifests['train'], role='train')
        dev = data.load_manifest(self.manifests['dev'], role='dev'); assert_disjoint(train, dev)
        with self.assertRaises(ValueError): assert_disjoint(train, train)
        dev[0]['group_id'] = train[0]['group_id']
        with self.assertRaises(ValueError): assert_disjoint(train, dev)

    def test_16_benchmark_test_not_train_or_dev(self):
        raw = dict(self.rows['test'][0], benchmark_id='mosaic_benchmark_6000_v1')
        p = write_csv(self.p/'benchmark.csv', [raw])
        for role in ['train', 'dev']:
            with self.assertRaises(ValueError): data.load_manifest(p, role=role)

    def test_17_prediction_count_and_nan_rejected(self):
        from torch.utils.data import DataLoader
        fixtures = [dict(image=torch.zeros(2), label=torch.tensor(float(i)), manifest_index=torch.tensor(i),
                         sample_id=str(i), sha256='', domain='Chart') for i in range(2)]
        class Detector(torch.nn.Module):
            def __init__(self, short): super().__init__(); self.short = short
            def forward(self, x): return torch.full((len(x)-int(self.short),), float('nan') if not self.short else 0.), x
        for short in [True, False]:
            with self.assertRaises((ValueError, FloatingPointError)):
                model.collect_predictions(Detector(short), DataLoader(fixtures, batch_size=2), device=torch.device('cpu'), amp_enabled=False)

    def test_18_strict_checkpoint_and_adamw_restore(self):
        parent, rows = self.fake_parent()
        with patch.object(checkpoint, 'build_detector', tiny_detector):
            detector, optimizer, _, _ = checkpoint.restore_detector(parent, device=torch.device('cpu'), restore_optimizer=True,
                require_reference=True, expected_config=self.config,
                train_manifest_sha256=file_sha256(self.manifests['train']), train_content_sha256=content_sha256(rows))
        self.assertTrue(all(s['step'].item()==100 for s in optimizer.state.values()))

    def test_19_wrong_parent_source_and_state_rejected(self):
        parent, rows = self.fake_parent(); base = checkpoint.load_checkpoint(parent)
        for kind in ['stage', 'manifest', 'missing_key', 'nan']:
            item = copy.deepcopy(base)
            if kind=='stage': item['metadata']['stage']='visual_expert'
            if kind=='manifest': item['metadata']['train_manifest_sha256']='c'*64
            if kind=='missing_key': item['model_state_dict'].pop('head.bias')
            if kind=='nan': item['model_state_dict']['head.bias'].fill_(float('nan'))
            p = self.p/f'bad_{kind}.pth'; torch.save(item, p)
            with patch.object(checkpoint, 'build_detector', tiny_detector), self.assertRaises((ValueError, RuntimeError, FloatingPointError)):
                checkpoint.restore_detector(p, device=torch.device('cpu'), restore_optimizer=True, require_reference=True,
                    expected_config=self.config, train_manifest_sha256=file_sha256(self.manifests['train']), train_content_sha256=content_sha256(rows))

    def test_20_no_random_backbone_fallback_when_openclip_missing(self):
        with patch.dict(sys.modules, {'open_clip': None}), self.assertRaises(RuntimeError):
            model.build_detector(model_name='ViT-L-14-quickgelu', pretrained='openai', image_size=224,
                trainable_blocks=2, device=torch.device('cpu'))

    def test_21_visual_evaluator_refuses_implicit_overwrite(self):
        output = self.p/'evaluation'; output.mkdir(); (output/'metrics.json').write_text('{}')
        with self.assertRaises(FileExistsError): self.run_cli('evaluate_visual_expert.py', ['--device', 'cpu', '--output-dir', output])

    def test_23_evaluation_accepts_generator_domains_but_training_rejects(self):
        rows = [dict(r, domain='BigGAN') for r in self.rows['test'][:2]]
        path = write_csv(self.p/'generator.csv', rows)
        evaluated = data.load_manifest(path, require_domain=False)
        self.assertEqual({r['domain'] for r in evaluated}, {'BigGAN'})
        with self.assertRaises(ValueError): data.load_manifest(path, require_domain=True)

    def test_24_reference_dry_run_never_constructs_backbone(self):
        output = self.p/'must_not_exist.pth'
        with patch.object(model, 'build_detector', side_effect=AssertionError('dry-run constructed a model')):
            self.run_cli('train_reference.py', ['--train-manifest', self.manifests['train'],
                '--dev-manifest', self.manifests['dev'], '--output', output,
                '--device', 'cpu', '--dry-run'])
        self.assertFalse(output.exists())

    def test_22_whole_repository_synthetic_cli_pipeline(self):
        """Execute root CLI chain with tiny interfaces, never actual CLIP/Qwen."""
        from decd.mllm_expert import backbone
        parent=self.p/'reference.pth'; profile=self.p/'risk.json'; final=self.p/'visual.pth'
        common=['--train-manifest',self.manifests['train'],'--dev-manifest',self.manifests['dev'],'--device','cpu']
        with patch.object(model,'build_detector',tiny_detector), patch.object(checkpoint,'build_detector',tiny_detector):
            self.run_cli('train_reference.py',common+['--output',parent])
            self.run_cli('build_risk_profile.py',['--manifest',self.manifests['train'],'--checkpoint',parent,'--output',profile,'--device','cpu'])
            self.run_cli('train_visual_expert.py',common+['--reference-checkpoint',parent,'--risk-profile',profile,'--output',final])
            for split in self.rows:
                self.run_cli('evaluate_visual_expert.py',['--manifest',self.manifests[split],'--checkpoint',final,'--output-dir',self.p/f'cv_{split}','--device','cpu'])
        result=checkpoint.load_checkpoint(final)
        self.assertTrue(all(v['step'].item()==150 for v in result['optimizer_state_dict']['state'].values()))
        mlcfg=yaml.safe_load((ROOT/'configs/mllm_expert_reported.yaml').read_text())
        mlcfg.update(feature_dim=8,epochs=1)
        config_path=self.p/'mllm.yaml'; config_path.write_text(yaml.safe_dump(mlcfg))
        with patch.object(backbone,'Qwen25VLBackbone',TinyFrozenBackbone):
            for split in self.rows:
                self.run_cli('extract_mllm_features.py',['--config',config_path,'--manifest',self.manifests[split],
                    '--output',self.p/f'{split}_features.pt','--device','cpu'])
        mllm=self.p/'mllm.pth'
        self.run_cli('train_mllm_expert.py',['--config',config_path,'--train-features',self.p/'train_features.pt',
            '--dev-features',self.p/'dev_features.pt','--output',mllm,'--device','cpu','--epochs',1])
        for split in self.rows:
            self.run_cli('build_dispatcher_signals.py',['--features',self.p/f'{split}_features.pt','--mllm-checkpoint',mllm,
                '--cv-predictions',self.p/f'cv_{split}/predictions.csv','--output',self.p/f'{split}_signals.csv','--device','cpu'])
        gate=self.p/'dispatcher.pth'
        self.run_cli('train_dispatcher.py',['--train-signals',self.p/'train_signals.csv','--dev-signals',self.p/'dev_signals.csv',
            '--output',gate,'--epochs',1,'--device','cpu'])
        self.run_cli('evaluate_dispatcher.py',['--signals',self.p/'test_signals.csv','--checkpoint',gate,
            '--output-dir',self.p/'fused_eval','--device','cpu'])
        metrics=json.loads((self.p/'fused_eval/metrics.json').read_text())
        self.assertEqual(metrics['n'],6)
        self.assertTrue(0 <= metrics['accuracy'] <= 1)
        self.assertLess(final.stat().st_size,100000)
        self.assertFalse(torch.cuda.is_initialized())


class KnownNonVisualIssues(Fixtures):
    @unittest.expectedFailure
    def test_mllm_same_cache_should_not_be_train_and_dev(self):
        from decd.mllm_expert.features import save_feature_cache
        from decd.mllm_expert.prompts import REASONING_PROMPTS
        cache=self.p/'same_train_dev.pt'
        save_feature_cache(cache,dict(features=torch.ones(2,3,8),labels=torch.tensor([0.,1.]),
            sample_ids=['a','b'],domains=['Chart','Chart'],prompts=list(REASONING_PROMPTS),feature_dim=8))
        cfg=yaml.safe_load((ROOT/'configs/mllm_expert_reported.yaml').read_text());cfg.update(feature_dim=8,epochs=1)
        config=self.p/'same_cache_config.yaml';config.write_text(yaml.safe_dump(cfg))
        with self.assertRaises(ValueError):
            self.run_cli('train_mllm_expert.py',['--config',config,'--train-features',cache,
                '--dev-features',cache,'--output',self.p/'same_cache_expert.pth','--device','cpu','--epochs',1])

    @unittest.expectedFailure
    def test_dispatcher_ties_should_meet_rejection_budget(self):
        from decd.dispatcher.abstention import abstention_threshold
        confidence=np.array([.2,.2,.2,.9]); tau=abstention_threshold(confidence,.5)
        self.assertGreaterEqual(np.mean(confidence < tau), .5)

    @unittest.expectedFailure
    def test_dispatcher_domain_should_use_requested_decision_threshold(self):
        from decd.dispatcher.evaluation import evaluate_dispatcher
        gate=torch.nn.Linear(6,1);gate.weight.data.zero_();gate.bias.data.zero_()
        signals=dict(labels=np.array([0,1]), cv_logits=np.array([.4,.4]),
            mllm_logits=np.full((2,3),.4),domains=['RealDomain','FakeDomain'])
        result=evaluate_dispatcher(gate,signals,temperature_cv=1.,temperature_mllm=1.,
            abstention_threshold_value=0.,decision_threshold=.7)
        self.assertEqual(result['per_domain']['RealDomain']['accuracy'],1.)

    @unittest.expectedFailure
    def test_mllm_feature_cache_should_reject_nan(self):
        from decd.mllm_expert.features import save_feature_cache
        payload=dict(features=torch.full((2,3,8),float('nan')),labels=torch.tensor([0.,1.]),
            sample_ids=['a','b'],prompts=['p1','p2','p3'],feature_dim=8)
        with self.assertRaises(ValueError):save_feature_cache(self.p/'nan.pt',payload)

    @unittest.expectedFailure
    def test_signal_join_should_reject_conflicting_cv_labels(self):
        from decd.mllm_expert.model import build_semantic_expert
        from decd.mllm_expert.checkpoint import save_expert_checkpoint
        from decd.mllm_expert.features import save_feature_cache
        from decd.mllm_expert.prompts import REASONING_PROMPTS
        expert,_=build_semantic_expert(feature_dim=8)
        mllm=self.p/'join_mllm.pth'; save_expert_checkpoint(mllm,expert=expert,config={},prompts=REASONING_PROMPTS)
        cache=self.p/'join_features.pt'
        save_feature_cache(cache,dict(features=torch.ones(2,3,8),labels=torch.tensor([0.,1.]),
            sample_ids=['a','b'],domains=['Chart','Chart'],prompts=list(REASONING_PROMPTS),feature_dim=8))
        scores=write_csv(self.p/'conflict.csv',[dict(sample_id='a',label=1,logit=-1.),dict(sample_id='b',label=0,logit=1.)])
        with self.assertRaises(ValueError):
            self.run_cli('build_dispatcher_signals.py',['--features',cache,'--mllm-checkpoint',mllm,
                '--cv-predictions',scores,'--output',self.p/'joined.csv','--device','cpu'])


if __name__=='__main__': unittest.main(verbosity=2)
