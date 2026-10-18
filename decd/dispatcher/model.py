"""Two-layer routing MLP, calibrated convex fusion, and routed confidence."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .signals import RELIABILITY_DIM


class ConfidenceAwareDispatcher(nn.Module):
    """``G_eta``: 6 -> 64 (ReLU) -> 1 gate logit; experts are never inputs."""

    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        self.hidden = nn.Linear(RELIABILITY_DIM, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, reliability: torch.Tensor) -> torch.Tensor:
        if reliability.shape[-1] != RELIABILITY_DIM:
            raise ValueError(
                f"expected {RELIABILITY_DIM}-d reliability vectors, "
                f"got {reliability.shape[-1]}-d"
            )
        return self.output(F.relu(self.hidden(reliability))).squeeze(-1)


def gate_weights(
    dispatcher: ConfidenceAwareDispatcher,
    reliability: torch.Tensor,
) -> torch.Tensor:
    return torch.sigmoid(dispatcher(reliability))


def fused_probability(
    gate_logits: torch.Tensor,
    cv_probabilities: torch.Tensor,
    mllm_probabilities: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``p = w p_cv + (1 - w) p_mllm``; returns ``(p, w)``."""

    w = torch.sigmoid(gate_logits)
    fused = w * cv_probabilities + (1.0 - w) * mllm_probabilities
    return fused, w


def routed_confidence(weights: torch.Tensor, fused: torch.Tensor) -> torch.Tensor:
    """``delta = max(w, 1-w) * 2 |p - 0.5|``."""

    decisiveness = torch.maximum(weights, 1.0 - weights)
    decision_margin = 2.0 * torch.abs(fused - 0.5)
    return decisiveness * decision_margin


def dispatcher_loss(
    gate_logits: torch.Tensor,
    cv_probabilities: torch.Tensor,
    mllm_probabilities: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """BCE on the calibrated convex combination (only the gate is learned)."""

    fused, _ = fused_probability(
        gate_logits,
        cv_probabilities,
        mllm_probabilities,
    )
    return F.binary_cross_entropy(fused.clamp(1.0e-7, 1.0 - 1.0e-7), labels.float())
