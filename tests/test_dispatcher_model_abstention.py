"""Tests for the routing MLP, convex fusion, confidence, and abstention."""

import numpy as np
import pytest
import torch

from decd.dispatcher.abstention import (
    UNCERTAIN,
    abstention_threshold,
    selective_predictions,
)
from decd.dispatcher.model import (
    ConfidenceAwareDispatcher,
    dispatcher_loss,
    fused_probability,
    gate_weights,
    routed_confidence,
)


def test_dispatcher_shapes_and_gate_range():
    torch.manual_seed(0)
    module = ConfidenceAwareDispatcher(hidden_dim=64)
    reliability = torch.rand(10, 6)
    logits = module(reliability)
    weights = gate_weights(module, reliability)
    assert logits.shape == (10,)
    assert torch.all(weights > 0) and torch.all(weights < 1)
    # 64 hidden units plus the output layer; ~0.5K parameters.
    assert sum(p.numel() for p in module.parameters()) == 6 * 64 + 64 + 64 + 1


def test_fused_probability_is_convex_combination():
    gate_logits = torch.tensor([10.0, -10.0, 0.0])
    p_cv = torch.full((3,), 0.8)
    p_mllm = torch.full((3,), 0.2)
    fused, weights = fused_probability(gate_logits, p_cv, p_mllm)
    assert torch.allclose(fused[0], p_cv[0], atol=1e-4)  # gate -> 1: CV
    assert torch.allclose(fused[1], p_mllm[1], atol=1e-4)  # gate -> 0: MLLM
    assert torch.allclose(fused[2], torch.tensor(0.5), atol=1e-4)
    assert torch.all((fused >= 0) & (fused <= 1))
    assert torch.allclose(weights, torch.tensor([1.0, 0.0, 0.5]), atol=1e-4)


def test_routed_confidence_hand_value():
    weights = torch.tensor([0.9, 0.5])
    fused = torch.tensor([0.8, 0.5])
    confidence = routed_confidence(weights, fused)
    expected = torch.tensor([0.9 * 0.6, 0.5 * 0.0])
    assert torch.allclose(confidence, expected)


def test_dispatcher_loss_updates_only_gate():
    torch.manual_seed(1)
    module = ConfidenceAwareDispatcher()
    reliability = torch.rand(16, 6)
    p_cv = torch.rand(16)
    p_mllm = torch.rand(16)
    labels = torch.randint(0, 2, (16,)).float()
    before = module.output.weight.detach().clone()
    loss = dispatcher_loss(module(reliability), p_cv, p_mllm, labels)
    assert torch.isfinite(loss)
    loss.backward()
    optimizer = torch.optim.AdamW(module.parameters(), lr=1e-2)
    optimizer.step()
    assert not torch.allclose(module.output.weight, before)
    # Expert signals are plain inputs and accumulate no gradients.
    assert p_cv.grad is None and p_mllm.grad is None


def test_abstention_threshold_matches_inf_definition():
    values = np.arange(100, dtype=np.float64)
    tau = abstention_threshold(values, budget=0.15)
    assert tau == 15.0
    assert np.mean(values < tau) >= 0.15
    # Zero rejection budget disables abstention entirely.
    assert abstention_threshold(values, budget=0.0) == 0.0


def test_abstention_threshold_infimum_with_ties():
    values = np.array([0.1, 0.1, 0.1, 0.5, 0.9])
    tau = abstention_threshold(values, budget=0.2)
    # Infimum is the tied value itself; strict comparison defers "up to ties".
    assert tau == pytest.approx(0.1)


def test_abstention_threshold_validation():
    with pytest.raises(ValueError):
        abstention_threshold(np.arange(5.0), budget=1.0)
    with pytest.raises(ValueError):
        abstention_threshold(np.arange(5.0), budget=-0.1)
    with pytest.raises(ValueError):
        abstention_threshold(np.array([], dtype=np.float64), budget=0.15)


def test_selective_predictions():
    probabilities = np.array([0.9, 0.1, 0.7])
    confidence = np.array([0.8, 0.2, 0.6])
    predictions = selective_predictions(
        probabilities,
        confidence,
        threshold=0.5,
    )
    assert predictions.tolist() == [1, UNCERTAIN, 1]
