"""Detection and reported real-only selective margin repair losses."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def require_finite(value: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"non-finite {name}")


def detection_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    require_finite(logits, 'logits')
    require_finite(labels, 'labels')
    if logits.shape != labels.shape or logits.ndim != 1 or not logits.numel():
        raise ValueError('logits/labels must be matching nonempty vectors')
    if not bool(((labels == 0) | (labels == 1)).all()):
        raise ValueError('binary labels must be 0 or 1')
    loss = F.binary_cross_entropy_with_logits(logits, labels.float())
    require_finite(loss, 'detection loss')
    return loss


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

    require_finite(logits, 'repair logits')
    require_finite(labels, 'repair labels')
    require_finite(weight_matrix, 'repair weights')
    if not math.isfinite(gamma) or gamma < 0:
        raise ValueError('gamma must be finite and nonnegative')
    if logits.shape != labels.shape or domain_ids.shape != labels.shape:
        raise ValueError('repair vectors must have matching shapes')
    if weight_matrix.shape != (3, 2) or bool((weight_matrix < 0).any()):
        raise ValueError('invalid six-cell repair weights')
    if not bool(((labels == 0) | (labels == 1)).all()):
        raise ValueError('invalid repair label')
    if not bool(((domain_ids >= 0) & (domain_ids < 3) & (domain_ids == domain_ids.long())).all()):
        raise ValueError('invalid domain ID')
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
    require_finite(loss, 'repair loss')

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
    require_finite(loss, 'loss before autograd')
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        create_graph=False,
        allow_unused=True,
    )
    gradient_l2_norm(gradients)
    return [
        None if gradient is None else gradient.detach()
        for gradient in gradients
    ]


def gradient_l2_norm(gradients: Sequence[torch.Tensor | None]) -> float:
    squared = 0.0
    for gradient in gradients:
        if gradient is not None:
            require_finite(gradient, 'gradient')
            squared += float(gradient.double().square().sum().item())
    if not math.isfinite(squared):
        raise FloatingPointError('non-finite squared gradient norm')
    return math.sqrt(max(squared, 0.0))


def adaptive_auxiliary_scale(
    detection_gradients: Sequence[torch.Tensor | None],
    repair_gradients: Sequence[torch.Tensor | None],
    *,
    rho_max: float,
    epsilon: float,
) -> tuple[float, dict[str, float]]:
    """Compute rho = min(||g_det|| / (||g_rep|| + eps), rho_max)."""

    if not math.isfinite(rho_max) or rho_max <= 0:
        raise ValueError("rho_max must be positive")
    if not math.isfinite(epsilon) or epsilon <= 0:
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

    if not len(parameters) == len(detection_gradients) == len(repair_gradients):
        raise ValueError('gradient list length mismatch')
    if not all(math.isfinite(x) and x >= 0 for x in (repair_lambda, rho)):
        raise ValueError('invalid auxiliary multiplier')
    gradient_l2_norm(detection_gradients)
    gradient_l2_norm(repair_gradients)
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
            if parameter.shape != final_gradient.shape:
                raise ValueError('gradient/parameter shape mismatch')
            require_finite(final_gradient, 'combined gradient')
            final_squared += float(
                final_gradient.double().square().sum().item()
            )
    if not math.isfinite(final_squared):
        raise FloatingPointError('non-finite combined gradient norm')
    return math.sqrt(max(final_squared, 0.0))


def linear_ramp(batch_index: int, ramp_batches: int) -> float:
    if batch_index <= 0:
        raise ValueError("batch_index is one-based and must be positive")
    if ramp_batches < 0:
        raise ValueError("ramp_batches must be non-negative")
    return 1.0 if ramp_batches == 0 else min(1.0, batch_index / ramp_batches)
