import math
from pathlib import Path
import runpy
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from decd.visual_expert.sampler import DomainLabelBalancedBatchSampler, EXPECTED_CELLS
from decd.visual_expert.losses import (detection_loss, real_only_margin_repair, compute_gradients,
    gradient_l2_norm, adaptive_auxiliary_scale, assign_combined_gradients, linear_ramp)
from decd.visual_expert.risk import build_risk_profile, summarize_margins, validate_risk_profile, _normalize_labelwise
from decd.visual_expert.model import collect_predictions, binary_metrics, configure_trainable_parameters, CLIPBinaryDetector


@pytest.mark.parametrize('batches', [100, 50])
def test_fixed_minibatches(batches):
    rows = [dict(domain=d, label=y) for d, y in EXPECTED_CELLS for _ in range(4)]
    batches1 = list(DomainLabelBalancedBatchSampler(rows, 6, batches, 2026))
    assert len(batches1) == batches
    assert batches1 == list(DomainLabelBalancedBatchSampler(rows, 6, batches, 2026))
    assert all(sorted((rows[i]['domain'], rows[i]['label']) for i in b) == sorted(EXPECTED_CELLS) for b in batches1)


@pytest.mark.parametrize('batch_size,num_batches', [(0, 100), (7, 100), (6, 0)])
def test_sampler_rejects(batch_size, num_batches):
    with pytest.raises(ValueError): DomainLabelBalancedBatchSampler([], batch_size, num_batches, 2026)


def test_mean_vs_squared_hinge():
    result = summarize_margins(np.array([0., 1.]), np.array([0, 0]), .5)
    assert result['risk_mean_m0p5'] == 1.
    assert result['risk_sq_mean_m0p5'] == 1.25


def test_normalization_preserves_historical_tie_policy():
    raw = np.array([[0., 1.], [1., 3.], [2., 5.]])
    np.testing.assert_allclose(_normalize_labelwise(raw, mode='minmax', alpha=1.), [[0, 0], [.5, .5], [1, 1]])
    np.testing.assert_array_equal(_normalize_labelwise(np.zeros((3, 2)), mode='minmax', alpha=1.), np.ones((3, 2)))


def test_real_only_and_actual_gradients():
    z = torch.tensor([.2, .3], requires_grad=True)
    loss, _ = real_only_margin_repair(logits=z, labels=torch.tensor([0., 1.]), domain_ids=torch.tensor([0, 0]), weight_matrix=torch.ones(3, 2), gamma=.5)
    g = compute_gradients(loss, [z], retain_graph=False)[0]
    torch.testing.assert_close(loss, torch.tensor(.49))
    torch.testing.assert_close(g, torch.tensor([1.4, 0.]))


def test_rho_ramp_and_combination():
    rho, _ = adaptive_auxiliary_scale([torch.tensor([3.])], [torch.tensor([1.])], rho_max=2., epsilon=1e-30)
    assert rho == 2.
    assert [linear_ramp(i, 10) for i in [1, 10, 50]] == [.1, 1., 1.]
    p = torch.nn.Parameter(torch.tensor([1.]))
    assign_combined_gradients(parameters=[p], detection_gradients=[torch.tensor([2.])], repair_gradients=[torch.tensor([3.])], repair_lambda=.1, rho=2.)
    torch.testing.assert_close(p.grad, torch.tensor([2.6]))


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_loss_and_gradient_rejected(bad):
    with pytest.raises(FloatingPointError): detection_loss(torch.tensor([bad]), torch.tensor([0.]))
    with pytest.raises(FloatingPointError): gradient_l2_norm([torch.tensor([bad])])
    with pytest.raises(ValueError): binary_metrics(np.array([0, 1]), np.array([bad, .8]), .5)


def test_gradient_length_rejected():
    with pytest.raises(ValueError): assign_combined_gradients(parameters=[], detection_gradients=[torch.ones(1)], repair_gradients=[], repair_lambda=.1, rho=1.)


@pytest.mark.parametrize('change', ['manifest', 'reference', 'gamma', 'alpha'])
def test_profile_provenance_rejected(change):
    rows = [dict(domain=d, label=y, logit=float(i)) for i, (d, y) in enumerate(EXPECTED_CELLS)]
    profile = build_risk_profile(rows, gamma=.5, risk_column='risk_mean_m0p5', normalization='minmax', alpha=1., provenance={'manifest_sha256':'a'*64, 'reference_checkpoint_sha256':'b'*64})
    kw = dict(risk_column='risk_mean_m0p5', gamma=.5, normalization='minmax', normalization_alpha=1., train_manifest_sha256='a'*64, reference_checkpoint_sha256='b'*64)
    validate_risk_profile(profile, **kw)
    field = {'manifest':'train_manifest_sha256', 'reference':'reference_checkpoint_sha256', 'gamma':'gamma', 'alpha':'normalization_alpha'}[change]
    kw[field] = 'c'*64 if change in ['manifest','reference'] else 2.
    with pytest.raises(ValueError): validate_risk_profile(profile, **kw)


@pytest.mark.parametrize('mode', ['short', 'nan', 'reverse'])
def test_prediction_contract(mode):
    fixtures = [dict(image=torch.zeros(2), label=torch.tensor(float(i)), manifest_index=torch.tensor(1-i if mode=='reverse' else i), sample_id=str(i), sha256='', domain='Chart') for i in range(2)]
    class Detector(torch.nn.Module):
        def forward(self, x):
            return torch.full((len(x)-int(mode=='short'),), float('nan') if mode=='nan' else 0.), x
    with pytest.raises((ValueError, FloatingPointError)):
        collect_predictions(Detector(), DataLoader(fixtures, batch_size=2), device=torch.device('cpu'), amp_enabled=False)


def test_valid_predictions_metrics():
    metrics = binary_metrics(np.array([0, 1]), np.array([.1, .9]), .5)
    assert all(metrics[k] == 1. for k in ['accuracy','balanced_accuracy','f1','auc','ap'])


def test_config_unchanged():
    import yaml
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root/'configs/visual_expert_reported.yaml').read_text())
    assert {k:config[k] for k in ['seed','backbone_lr','head_lr','weight_decay','amp','gamma','repair_lambda','rho_max','ramp_batches','risk_column']} == dict(seed=2026,backbone_lr=1e-5,head_lr=1e-4,weight_decay=1e-4,amp=True,gamma=.5,repair_lambda=.1,rho_max=10.,ramp_batches=10,risk_column='risk_mean_m0p5')
