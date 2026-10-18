"""Detection and reported real-only selective margin repair losses."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def detection_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, labels.float())


def real_only_margin_repair(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    domain_ids: torch.Tensor,
    weight_matrix: torch.Tensor,
    gamma: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Risk-weighted squared hinge on real samples only.

    Real samples require ``logit <= -gamma``. Fake samples remain in the
    detection objective but receive no auxiliary repair weight.
    """

    labels_long = labels.long()
    real_mask = labels_long == 0
    signed = -logits
    deficits = F.relu(float(gamma) - signed)
    sample_weights = weight_matrix[
        domain_ids.long(),
        labels_long,
    ] * real_mask.to(logits.dtype)
    denominator = sample_weights.sum().clamp_min(1.0e-12)
    loss = (sample_weights * deficits.square()).sum() / denominator

    return loss, {
        "active_rate": (
            ((deficits > 0) & real_mask).float().sum()
            / real_mask.float().sum().clamp_min(1.0)
        ),
        "weighted_active_rate": (
            ((deficits > 0).float() * sample_weights).sum() / denominator
        ),
        "mean_real_deficit": (
            deficits[real_mask].mean()
            if bool(real_mask.any())
            else torch.zeros((), device=logits.device)
        ),
    }


def compute_gradients(
    loss: torch.Tensor,
    parameters: Sequence[nn.Parameter],
    *,
    retain_graph: bool,
) -> list[torch.Tensor | None]:
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        create_graph=False,
        allow_unused=True,
    )
    return [
        None if gradient is None else gradient.detach()
        for gradient in gradients
    ]


def gradient_l2_norm(gradients: Sequence[torch.Tensor | None]) -> float:
    squared = 0.0
    for gradient in gradients:
        if gradient is not None:
            squared += float(gradient.double().square().sum().item())
    return math.sqrt(max(squared, 0.0))


def adaptive_auxiliary_scale(
    detection_gradients: Sequence[torch.Tensor | None],
    repair_gradients: Sequence[torch.Tensor | None],
    *,
    rho_max: float,
    epsilon: float,
) -> tuple[float, dict[str, float]]:
    """Compute rho = min(||g_det|| / (||g_rep|| + eps), rho_max)."""

    if rho_max <= 0:
        raise ValueError("rho_max must be positive")
    if epsilon <= 0:
        raise ValueError("gradient epsilon must be positive")
    detection_norm = gradient_l2_norm(detection_gradients)
    repair_norm = gradient_l2_norm(repair_gradients)
    rho = min(detection_norm / (repair_norm + epsilon), float(rho_max))
    return rho, {
        "detection_gradient_norm": detection_norm,
        "repair_gradient_norm": repair_norm,
        "rho": rho,
    }


def assign_combined_gradients(
    *,
    parameters: Sequence[nn.Parameter],
    detection_gradients: Sequence[torch.Tensor | None],
    repair_gradients: Sequence[torch.Tensor | None],
    repair_lambda: float,
    rho: float,
) -> float:
    """Assign g_det + repair_lambda * rho * g_rep and return its norm."""

    final_squared = 0.0
    for parameter, detection_gradient, repair_gradient in zip(
        parameters,
        detection_gradients,
        repair_gradients,
    ):
        final_gradient: torch.Tensor | None = None
        if detection_gradient is not None:
            final_gradient = detection_gradient.clone()
        if repair_gradient is not None:
            auxiliary = float(repair_lambda) * float(rho) * repair_gradient
            final_gradient = (
                auxiliary
                if final_gradient is None
                else final_gradient + auxiliary
            )
        parameter.grad = final_gradient
        if final_gradient is not None:
            final_squared += float(
                final_gradient.double().square().sum().item()
            )
    return math.sqrt(max(final_squared, 0.0))


def linear_ramp(batch_index: int, ramp_batches: int) -> float:
    if batch_index <= 0:
        raise ValueError("batch_index is one-based and must be positive")
    if ramp_batches < 0:
        raise ValueError("ramp_batches must be non-negative")
    return 1.0 if ramp_batches == 0 else min(1.0, batch_index / ramp_batches)
