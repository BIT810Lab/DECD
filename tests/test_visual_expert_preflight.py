"""CPU checks for frozen inputs, continuous AdamW, resume and repair ramp."""
import argparse
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from decd.visual_expert import checkpoint
from decd.visual_expert.model import build_optimizer
from decd.visual_expert.protocol import (
    validate_resume_contract, validate_train_dev_integrity, validate_training_schedule,
)
from decd.visual_expert.sampler import EXPECTED_CELLS

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('alternating_cpu_test', ROOT/'scripts/train_visual_expert_alternating.py')
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)


@pytest.fixture
def config():
    return yaml.safe_load((ROOT/'configs/visual_expert_mosaic_v2_full.yaml').read_text())


class TinyDetector(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.head=torch.nn.Linear(3,1)

    def forward(self, images):
        features=images.mean((2,3))
        return self.head(features).squeeze(1), features


def make_info(model):
    return dict(backbone_parameters=[],head_parameters=list(model.parameters()),
                trainable_parameters=list(model.parameters()),feature_dim=3)


def batch():
    return dict(image=torch.ones(6,3,2,2), label=torch.tensor([0.,1.,0.,1.,0.,1.]),
                domain_id=torch.tensor([0,0,1,1,2,2]))


def reference_payload(config):
    detector=TinyDetector()
    optimizer=build_optimizer(make_info(detector),backbone_lr=config['backbone_lr'],
                              head_lr=config['head_lr'],weight_decay=config['weight_decay'])
    trainer._bce_update(detector=detector,optimizer=optimizer,scaler=trainer._new_scaler(False),
                        batch=batch(),device=torch.device('cpu'),amp_enabled=False,threshold=.5)
    for item in optimizer.state.values(): item['step'].fill_(700)
    payload=dict(config=copy.deepcopy(config),metadata=dict(stage='reference',updates=700,
        train_manifest_sha256='train',dev_manifest_sha256='dev'),
        optimizer_state_dict=optimizer.state_dict(),model_state_dict=detector.state_dict())
    return detector,optimizer,payload


@pytest.mark.parametrize('key,value',[('cycles',7),('reference_batches',100),('batch_size',12),
    ('repair_batches_per_cycle',100),('expected_train_per_cell',700.0)])
def test_wrong_schedule_rejected(config,key,value):
    config[key]=value
    with pytest.raises(ValueError,match='schedule requires'): validate_training_schedule(config)


def test_strict_reference_resume_contract(config):
    _,_,payload=reference_payload(config)
    assert validate_resume_contract(payload,config,train_sha256='train',dev_sha256='dev')==0


@pytest.mark.parametrize('problem',['manifest','config','optimizer_missing','step','moment','stage'])
def test_resume_provenance_or_state_rejected(config,problem):
    _,_,payload=reference_payload(config)
    if problem=='manifest': payload['metadata']['train_manifest_sha256']='other'
    elif problem=='config': payload['config']['bce_sampler_seed']+=1
    elif problem=='optimizer_missing': payload['optimizer_state_dict']=None
    elif problem=='step': next(iter(payload['optimizer_state_dict']['state'].values()))['step'].fill_(699)
    elif problem=='moment': next(iter(payload['optimizer_state_dict']['state'].values())).pop('exp_avg')
    else: payload['metadata']['stage']='legacy_b50'
    with pytest.raises(ValueError): validate_resume_contract(payload,config,train_sha256='train',dev_sha256='dev')


def test_cycle_resume_checks_common_reference_profile_and_updates(config):
    _,_,payload=reference_payload(config)
    payload['config']['completed_cycle']=1
    payload['metadata'].update(stage='alternating_visual_expert',completed_cycle=1,
        global_updates=850,reference_checkpoint_sha256='reference',risk_profile_sha256='risk')
    for item in payload['optimizer_state_dict']['state'].values(): item['step'].fill_(850)
    assert validate_resume_contract(payload,config,train_sha256='train',dev_sha256='dev',
        reference_sha256='reference',risk_profile_sha256='risk')==1
    with pytest.raises(ValueError,match='reference checkpoint SHA'):
        validate_resume_contract(payload,config,train_sha256='train',dev_sha256='dev',
            reference_sha256='other',risk_profile_sha256='risk')


def test_model_adamw_roundtrip_and_global_repair_ramp(config,tmp_path,monkeypatch):
    detector,optimizer,_=reference_payload(config)
    parent=tmp_path/'synthetic_reference.pth'
    checkpoint.save_checkpoint(parent,detector=detector,optimizer=optimizer,config=config,
        feature_dim=3,metadata=dict(stage='reference',updates=700))
    def factory(**kwargs):
        model=TinyDetector()
        return model,make_info(model)
    monkeypatch.setattr(checkpoint,'build_detector',factory)
    restored,adam,info,_=checkpoint.restore_detector(parent,device=torch.device('cpu'),restore_optimizer=True)
    for key,value in detector.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key],value,rtol=0,atol=0)
    for original,loaded in zip(optimizer.state.values(),adam.state.values()):
        for key in ('step','exp_avg','exp_avg_sq'):
            torch.testing.assert_close(original[key],loaded[key],rtol=0,atol=0)
    trainer._bce_update(detector=restored,optimizer=adam,scaler=trainer._new_scaler(False),
        batch=batch(),device=torch.device('cpu'),amp_enabled=False,threshold=.5)
    config['ramp_batches']=100
    stats=trainer._joint_update(detector=restored,optimizer=adam,
        trainable_parameters=info['trainable_parameters'],batch=batch(),weights=torch.ones(3,2),
        config=config,device=torch.device('cpu'),amp_enabled=False,repair_update=51)
    assert stats['repair_lambda']==pytest.approx(.1*.51)
    assert {int(s['step']) for s in adam.state.values()}=={702}


def test_nonfinite_update_fails_before_adamw_step(config):
    detector=TinyDetector(); optimizer=torch.optim.AdamW(detector.parameters())
    b=batch(); b['image'].fill_(float('nan'))
    with pytest.raises((FloatingPointError, ValueError)):
        trainer._bce_update(detector=detector,optimizer=optimizer,scaler=trainer._new_scaler(False),
            batch=b,device=torch.device('cpu'),amp_enabled=False,threshold=.5)
    assert not optimizer.state


def test_train_dev_actual_hash_role_and_group_checks(tmp_path):
    a,b=tmp_path/'a.png',tmp_path/'b.png'
    a.write_bytes(b'a'); b.write_bytes(b'b')
    train=[dict(image_path=a,sha256=checkpoint.file_sha256(a),split='train',group_id='one')]
    dev=[dict(image_path=b,sha256=checkpoint.file_sha256(b),split='dev',group_id='two')]
    assert validate_train_dev_integrity(train,dev)['verified_image_sha256']==2
    dev[0]['group_id']='one'
    with pytest.raises(ValueError,match='source group overlap'): validate_train_dev_integrity(train,dev)
    dev[0]['group_id']='two'; dev[0]['split']='test'
    with pytest.raises(ValueError,match='role mismatch'): validate_train_dev_integrity(train,dev)
    dev[0]['split']='dev'; a.write_bytes(b'tampered')
    with pytest.raises(ValueError,match='SHA256 mismatch'): validate_train_dev_integrity(train,dev)


def test_dry_run_forces_cpu_and_never_constructs_model(config,monkeypatch,capsys):
    args=argparse.Namespace(config=ROOT/'configs/visual_expert_mosaic_v2_full.yaml',
        device=None,dry_run=True,resume=False,overwrite=False)
    monkeypatch.setattr(trainer,'parse_args',lambda:args)
    monkeypatch.setattr(trainer,'load_yaml_config',lambda _:config)
    def forbidden(*args,**kwargs): raise AssertionError('GPU/model path reached in dry-run')
    monkeypatch.setattr(trainer.torch.cuda,'is_available',forbidden)
    monkeypatch.setattr(trainer,'build_detector',forbidden)
    monkeypatch.setattr(trainer,'set_determinism',forbidden)
    def rows_for(path,**kwargs):
        role='dev' if 'dev' in str(path) else 'train'
        count=100 if role=='dev' else 700
        return [dict(sample_id=f'{role}-{d}-{y}-{i}',domain=d,label=y)
                for d,y in EXPECTED_CELLS for i in range(count)]
    monkeypatch.setattr(trainer,'load_manifest',rows_for)
    monkeypatch.setattr(trainer,'validate_train_dev_integrity',lambda *args:{'mock':1})
    monkeypatch.setattr(trainer.shutil,'disk_usage',lambda _:SimpleNamespace(free=100*1024**3))
    monkeypatch.setattr(trainer,'build_training_loader',lambda *args,**kwargs:[batch()])
    monkeypatch.setattr(trainer,'build_eval_loader',lambda *args,**kwargs:[])
    trainer.main()
    text=capsys.readouterr().out
    assert 'dry-run passed' in text and 'reference_updates=700, cycles=14' in text
