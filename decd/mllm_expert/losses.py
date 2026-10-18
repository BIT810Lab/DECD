"""Semantic-expert objectives: prompt BCE and neighborhood preservation."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

LOG_EPSILON = 1.0e-12


def prompt_detection_loss(
    prompt_logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Mean BCE over images and prompts: ``1 / (B K_p)`` per the paper."""

    if prompt_logits.ndim != 2:
        raise ValueError("prompt_logits must have shape (B, K)")
    targets = labels.float().unsqueeze(1).expand_as(prompt_logits)
    return F.binary_cross_entropy_with_logits(prompt_logits, targets)


def _neighborhood_rows(
    representations: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Row-wise softmax distributions of pairwise cosine similarities.

    Args:
        representations: ``(B, K, d)``.
        temperature: neighborhood softmax temperature ``tau_np``.

    Returns:
        Softmax rows ``(K, B, B)`` (row ``i`` is the neighborhood distribution
        of sample ``i`` under prompt ``k``).
    """

    normalized = F.normalize(representations, p=2, dim=-1, eps=1.0e-8)
    similarity = torch.matmul(
        normalized.transpose(0, 1),
        normalized.transpose(0, 1).transpose(1, 2),
    )
    return F.softmax(similarity / float(temperature), dim=-1)


def neighborhood_preservation_loss(
    frozen: torch.Tensor,
    adapted: torch.Tensor,
    temperature: float = 0.1,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """``KL(stopgrad(q_0) || q_a)`` averaged over images and prompts.

    The frozen backbone is the teacher: its rows are detached, and only the
    adapter moves.
    """

    if temperature <= 0:
        raise ValueError("neighborhood temperature must be positive")
    if frozen.shape != adapted.shape:
        raise ValueError("frozen and adapted representations must share a shape")
    if frozen.shape[0] < 2:
        zero = frozen.sum() * 0.0
        return zero, {"mean_kl": 0.0}

    teacher = _neighborhood_rows(frozen, temperature).detach()
    student = _neighborhood_rows(adapted, temperature)
    kl_rows = (
        teacher * (teacher.clamp_min(LOG_EPSILON).log() - student.clamp_min(LOG_EPSILON).log())
    ).sum(dim=-1)
    loss = kl_rows.mean()
    return loss, {"mean_kl": float(loss.detach().item())}


def semantic_expert_loss(
    *,
    prompt_logits: torch.Tensor,
    labels: torch.Tensor,
    frozen: torch.Tensor,
    adapted: torch.Tensor,
    np_temperature: float,
    np_lambda: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """``L_mllm = L_det^mllm + lambda_np * L_np``."""

    if np_lambda < 0:
        raise ValueError("np_lambda must be non-negative")
    detection = prompt_detection_loss(prompt_logits, labels)
    preservation, diagnostics = neighborhood_preservation_loss(
        frozen,
        adapted,
        temperature=np_temperature,
    )
    total = detection + float(np_lambda) * preservation
    diagnostics.update(
        {
            "detection_loss": float(detection.detach().item()),
            "np_loss": float(preservation.detach().item()),
            "total_loss": float(total.detach().item()),
        }
    )
    return total, diagnostics
