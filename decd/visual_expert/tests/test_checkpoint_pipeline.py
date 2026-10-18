import copy
from pathlib import Path
import runpy
import sys
import pytest
import torch
import yaml

from conftest import csv_file
from decd.visual_expert.model import CLIPBinaryDetector, configure_trainable_parameters, build_optimizer
from decd.visual_expert.checkpoint import (save_checkpoint, load_checkpoint, restore_detector,
    validate_training_config, validate_reference_checkpoint)

ROOT=Path(__file__).resolve().parents[1]


class Visual(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer=torch.nn.Module()
        self.transformer.resblocks=torch.nn.ModuleList([torch.nn.Linear(3,3) for _ in range(4)])
        self.ln_post=torch.nn.LayerNorm(3)


class TinyClip(torch.nn.Module):
    def __init__(self):
        super().__init__();self.visual=Visual();self.other=torch.nn.Parameter(torch.ones(1))
    def encode_image(self,x,normalize=False):
        x=x.mean((2,3))
        for block in self.visual.transformer.resblocks:x=torch.tanh(block(x))
        return self.visual.ln_post(x)


def toy_factory(**kwargs):
    det=CLIPBinaryDetector(TinyClip(),3)
    info=configure_trainable_parameters(det,2);info['feature_dim']=3
    return det,info


def config():return yaml.safe_load((ROOT/'configs/visual_expert_reported.yaml').read_text())


def test_trainable_scope():
    det,info=toy_factory()
    assert len(info['trainable_parameter_names'])==8
    assert all(any(s in n for s in ['resblocks.2.','resblocks.3.','ln_post.','head.']) for n in info['trainable_parameter_names'])
    assert not det.clip_model.other.requires_grad


def test_pretrained_is_required_and_restore_never_downloads(monkeypatch):
    import types
    import decd.visual_expert.model as module
    calls=[]
    def create(*args,**kwargs):
        calls.append(kwargs); return TinyClip()
    monkeypatch.setitem(sys.modules,'open_clip',types.SimpleNamespace(create_model=create))
    for loading in [True,False]:
        module.build_detector(model_name='ViT-L-14-quickgelu',pretrained='openai',image_size=224,
                              trainable_blocks=2,device=torch.device('cpu'),load_pretrained=loading,feature_dim=3)
    assert calls[0]['require_pretrained'] and calls[0]['pretrained']=='openai'
    assert not calls[1]['require_pretrained'] and calls[1]['pretrained'] is None


def test_missing_checkpoint_fails(tmp_path):
    with pytest.raises(FileNotFoundError):load_checkpoint(tmp_path/'absent.pth')


def test_save_nonfinite_rejected(tmp_path):
    det,_=toy_factory()
    det.head.bias.data.fill_(float('nan'))
    with pytest.raises(FloatingPointError):
        save_checkpoint(tmp_path/'bad.pth',detector=det,optimizer=None,config=config(),feature_dim=3)
    assert not (tmp_path/'bad.pth').exists()


def payload(tmp_path):
    det,info=toy_factory();c=config()
    opt=build_optimizer(info,backbone_lr=c['backbone_lr'],head_lr=c['head_lr'],weight_decay=c['weight_decay'])
    det(torch.ones(2,3,2,2))[0].sum().backward();opt.step()
    for item in opt.state.values():item['step'].fill_(100)
    p=tmp_path/'toy.pth'
    save_checkpoint(p,detector=det,optimizer=opt,config=c,feature_dim=3,metadata=dict(stage='reference',completed_batches=100,train_manifest_sha256='a'*64,train_content_sha256='b'*64))
    return p,det,opt


def test_roundtrip_strict_model_and_AdamW(tmp_path,monkeypatch):
    p,det,opt=payload(tmp_path)
    monkeypatch.setattr('decd.visual_expert.checkpoint.build_detector',toy_factory)
    restored,optimizer,_,_=restore_detector(p,device=torch.device('cpu'),restore_optimizer=True,require_reference=True,expected_config=config(),train_manifest_sha256='a'*64,train_content_sha256='b'*64)
    torch.testing.assert_close(det(torch.ones(2,3,2,2))[0],restored(torch.ones(2,3,2,2))[0],rtol=0,atol=0)
    for a,b in zip(opt.state.values(),optimizer.state.values()):
        assert a.keys()==b.keys()
        for key in a:torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)


@pytest.mark.parametrize('kind',['stage','batches','manifest','content','seed','lr','step','missing_state','nan_state'])
def test_parent_rejections(tmp_path,monkeypatch,kind):
    p,_,_=payload(tmp_path);data=load_checkpoint(p)
    if kind=='stage':data['metadata']['stage']='visual_expert'
    if kind=='batches':data['metadata']['completed_batches']=2
    if kind=='manifest':data['metadata']['train_manifest_sha256']='c'*64
    if kind=='content':data['metadata']['train_content_sha256']='c'*64
    if kind=='seed':data['config']['seed']=2027
    if kind=='lr':data['optimizer_state_dict']['param_groups'][0]['lr']=.5
    if kind=='step':next(iter(data['optimizer_state_dict']['state'].values()))['step'].fill_(99)
    if kind=='missing_state':data['model_state_dict'].pop('head.bias')
    if kind=='nan_state':data['model_state_dict']['head.bias'].fill_(float('nan'))
    torch.save(data,p)
    monkeypatch.setattr('decd.visual_expert.checkpoint.build_detector',toy_factory)
    with pytest.raises((ValueError,RuntimeError,FloatingPointError)):
        restore_detector(p,device=torch.device('cpu'),restore_optimizer=True,require_reference=True,expected_config=config(),train_manifest_sha256='a'*64,train_content_sha256='b'*64)


@pytest.mark.parametrize('field,value',[('repair_label_side','fake'),('rho_max',float('nan')),('model_name','other'),('reference_batches',99),('batch_size',7)])
def test_config_rejections(field,value):
    c=config();c[field]=value
    with pytest.raises(ValueError):validate_training_config(c)


def run_script(monkeypatch,name,args):
    monkeypatch.setattr(sys,'argv',[name]+list(map(str,args)))
    runpy.run_path(str(ROOT/'scripts'/name),run_name='__main__')


@pytest.mark.parametrize('name,maximum',[('train_reference.py',100),('train_visual_expert.py',50)])
@pytest.mark.parametrize('kind',['zero','negative','above_endpoint'])
def test_cli_batch_bounds(monkeypatch,name,maximum,kind):
    value={'zero':0,'negative':-1,'above_endpoint':maximum+1}[kind]
    with pytest.raises(ValueError):
        run_script(monkeypatch,name,['--config',ROOT/'configs/visual_expert_reported.yaml','--max-batches',value,'--device','cpu'])


def test_synthetic_B100_risk_B50_pipeline(tiny_pool,monkeypatch):
    """Tiny synthetic CPU integration only; NOT a pretrained-model reproduction."""
    root,rows=tiny_pool
    train=csv_file(root/'train.csv',[r for r in rows if r['split']=='train'])
    dev=csv_file(root/'dev.csv',[r for r in rows if r['split']=='dev'])
    c=ROOT/'configs/visual_expert_reported.yaml';parent=root/'b100.pth';risk=root/'risk.json';final=root/'b50.pth'
    monkeypatch.setattr('decd.visual_expert.model.build_detector',toy_factory)
    monkeypatch.setattr('decd.visual_expert.checkpoint.build_detector',toy_factory)
    common=['--config',c,'--train-manifest',train,'--dev-manifest',dev,'--device','cpu']
    run_script(monkeypatch,'train_reference.py',common+['--output',parent])
    assert load_checkpoint(parent)['metadata']['completed_batches']==100
    run_script(monkeypatch,'build_risk_profile.py',['--config',c,'--manifest',train,'--checkpoint',parent,'--output',risk,'--device','cpu'])
    run_script(monkeypatch,'train_visual_expert.py',common+['--reference-checkpoint',parent,'--risk-profile',risk,'--output',final])
    result=load_checkpoint(final)
    assert result['metadata']['completed_batches']==50 and result['metadata']['optimizer_restored']
    assert all(v['step'].item()==150 for v in result['optimizer_state_dict']['state'].values())
    assert parent.stat().st_size<100000 and final.stat().st_size<100000
