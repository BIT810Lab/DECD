"""Tests for the consensus gate, gated adapter, and PP losses."""

import torch
import torch.nn.functional as F

from decd.mllm_expert.adapter import GatedBottleneckAdapter, consensus_gates
from decd.mllm_expert.losses import (
    neighborhood_preservation_loss,
    prompt_detection_loss,
    semantic_expert_loss,
)
from decd.mllm_expert.model import MLLMSemanticExpert, ensemble_summary


def test_consensus_gate_values():
    # Identical views -> cosine 1 -> gate 1.
    base = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    z_consensus = base.repeat(1, 3, 1)
    assert torch.allclose(consensus_gates(z_consensus), torch.ones(1, 3))

    # Mutually orthogonal views -> mean cosine 0 -> gate 0.5.
    z_ortho = torch.eye(3).unsqueeze(0)
    assert torch.allclose(
        consensus_gates(z_ortho),
        torch.full((1, 3), 0.5),
        atol=1e-6,
    )

    # A lone opposite view has agreement -1 (gate 0); its peers average 0.
    z_opposed = torch.tensor(
        [[[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]]]
    )
    gates = consensus_gates(z_opposed)
    assert gates.shape == (1, 3)
    assert torch.allclose(gates[0, 2], torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(gates[0, 0], torch.tensor(0.5), atol=1e-6)
    assert float(gates.min()) >= 0.0 and float(gates.max()) <= 1.0


def test_single_view_gate_is_one():
    z = torch.randn(5, 1, 8)
    assert torch.equal(consensus_gates(z), torch.ones(5, 1))


def test_zero_initialized_up_projection_starts_from_norm():
    torch.manual_seed(0)
    adapter = GatedBottleneckAdapter(16, bottleneck_ratio=4)
    assert torch.count_nonzero(adapter.up_projection.weight) == 0

    z0 = torch.randn(4, 3, 16)
    adapted, gates = adapter(z0)
    expected = F.layer_norm(z0, (16,), adapter.norm.weight, adapter.norm.bias)
    assert torch.allclose(adapted, expected, atol=1e-6)
    assert gates.shape == (4, 3)


def test_adapter_gate_has_no_parameters_and_is_detached():
    adapter = GatedBottleneckAdapter(16)
    z0 = torch.randn(4, 3, 16, requires_grad=True)
    adapted, gates = adapter(z0)
    loss = adapted.square().sum()
    loss.backward()
    assert not gates.requires_grad
    # Down/up projections receive gradients; gate path does not require them.
    assert adapter.down_projection.weight.grad is not None
    assert adapter.up_projection.weight.grad is not None


def test_expert_forward_shapes_and_parameter_budget():
    expert = MLLMSemanticExpert(feature_dim=32, num_prompts=3, bottleneck_ratio=4)
    outputs = expert(torch.randn(7, 3, 32))
    assert outputs["prompt_logits"].shape == (7, 3)
    assert outputs["prompt_probabilities"].shape == (7, 3)
    assert outputs["adapted"].shape == (7, 3, 32)
    assert outputs["gates"].shape == (7, 3)

    # d=3584 -> 6.43M trainable parameters, matching the paper's Table cost.
    expert_reported = MLLMSemanticExpert(3584, 3, 4)
    count = sum(p.numel() for p in expert_reported.parameters())
    assert count == 6_433_281


def test_prompt_detection_loss_matches_manual_bce():
    torch.manual_seed(0)
    logits = torch.randn(6, 3)
    labels = torch.tensor([0.0, 1.0, 0.0, 0.0, 1.0, 1.0])
    expected = F.binary_cross_entropy_with_logits(
        logits,
        labels.unsqueeze(1).expand_as(logits),
    )
    assert torch.allclose(prompt_detection_loss(logits, labels), expected)


def test_neighborhood_preservation_identical_is_zero():
    z0 = torch.randn(8, 3, 16)
    loss, diagnostics = neighborhood_preservation_loss(z0, z0, temperature=0.1)
    assert float(loss) < 1e-10
    assert diagnostics["mean_kl"] < 1e-10


def test_neighborhood_loss_nonnegative_and_teacher_frozen():
    torch.manual_seed(1)
    teacher = torch.randn(8, 3, 16, requires_grad=True)
    student = torch.randn(8, 3, 16, requires_grad=True)
    loss, _ = neighborhood_preservation_loss(teacher, student, temperature=0.1)
    assert float(loss) >= 0.0
    loss.backward()
    # KL(stopgrad(q0) || qa): only the adapter/student receives gradients.
    assert teacher.grad is None
    assert student.grad is not None


def test_single_sample_neighborhood_is_safe():
    z = torch.randn(1, 3, 8)
    loss, _ = neighborhood_preservation_loss(z, z, temperature=0.1)
    assert float(loss) == 0.0


def test_combined_loss_decomposes():
    torch.manual_seed(2)
    expert = MLLMSemanticExpert(16, 3, 4)
    z0 = torch.randn(8, 3, 16)
    labels = torch.randint(0, 2, (8,)).float()
    outputs = expert(z0)
    total, diagnostics = semantic_expert_loss(
        prompt_logits=outputs["prompt_logits"],
        labels=labels,
        frozen=outputs["frozen"],
        adapted=outputs["adapted"],
        np_temperature=0.1,
        np_lambda=0.5,
    )
    expected = diagnostics["detection_loss"] + 0.5 * diagnostics["np_loss"]
    assert abs(total.item() - expected) < 1e-6


def test_ensemble_summary():
    probs = torch.tensor([[0.2, 0.4, 0.6], [0.0, 0.5, 1.0]])
    mean, variance = ensemble_summary(probs)
    assert torch.allclose(mean, torch.tensor([0.4, 0.5]))
    expected_var = torch.tensor(
        [((0.2 - 0.4) ** 2 + (0.4 - 0.4) ** 2 + (0.6 - 0.4) ** 2) / 3, 1.0 / 6.0]
    )
    assert torch.allclose(variance, expected_var)
