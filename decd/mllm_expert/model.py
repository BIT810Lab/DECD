"""Prior-preserved semantic expert: PP adapter plus a binary prompt head.

The expert consumes frozen pooled MLLM representations ``z0`` of shape
``(B, K_p, d)``. The MLLM backbone itself never enters this module, which keeps
it backbone-agnostic, trivially testable on CPU, and compatible with the
precomputed-feature workflow (the frozen representations and gates are
constant throughout training).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn

from .adapter import GatedBottleneckAdapter, consensus_gates
from .prompts import REASONING_PROMPTS, validate_prompts


class MLLMSemanticExpert(nn.Module):
    """Gated bottleneck adapter and one shared linear fake/real head."""

    def __init__(
        self,
        feature_dim: int,
        num_prompts: int = len(REASONING_PROMPTS),
        bottleneck_ratio: int = 4,
    ) -> None:
        super().__init__()
        if num_prompts <= 0:
            raise ValueError("num_prompts must be positive")
        self.feature_dim = int(feature_dim)
        self.num_prompts = int(num_prompts)
        self.adapter = GatedBottleneckAdapter(feature_dim, bottleneck_ratio)
        self.head = nn.Linear(feature_dim, 1)

    def forward(
        self,
        frozen_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Run frozen features through the gate, adapter, and head.

        Returns:
            Mapping with ``prompt_logits`` ``(B, K)``, ``prompt_probabilities``
            ``(B, K)``, ``gates`` ``(B, K)``, ``adapted`` ``(B, K, d)``, and
            the detached ``frozen`` features.
        """

        if frozen_features.ndim != 3:
            raise ValueError("frozen_features must have shape (B, K, d)")
        if frozen_features.shape[-2] != self.num_prompts:
            raise ValueError(
                f"expected {self.num_prompts} prompt views, "
                f"got {frozen_features.shape[-2]}"
            )
        if frozen_features.shape[-1] != self.feature_dim:
            raise ValueError(
                f"expected feature dim {self.feature_dim}, "
                f"got {frozen_features.shape[-1]}"
            )
        gates = consensus_gates(frozen_features)
        adapted, gates = self.adapter(frozen_features, gates)
        logits = self.head(adapted).squeeze(-1)
        return {
            "prompt_logits": logits,
            "prompt_probabilities": torch.sigmoid(logits),
            "gates": gates,
            "adapted": adapted,
            "frozen": frozen_features,
        }


def build_semantic_expert(
    *,
    feature_dim: int,
    num_prompts: int = len(REASONING_PROMPTS),
    bottleneck_ratio: int = 4,
    device: torch.device | str = "cpu",
    prompts: Sequence[str] | None = None,
) -> tuple[MLLMSemanticExpert, dict[str, Any]]:
    """Construct the trainable expert and report its parameter budget."""

    selected_prompts = tuple(REASONING_PROMPTS if prompts is None else prompts)
    validate_prompts(selected_prompts)
    if len(selected_prompts) != num_prompts:
        raise ValueError("number of prompts does not match num_prompts")

    expert = MLLMSemanticExpert(
        feature_dim=feature_dim,
        num_prompts=num_prompts,
        bottleneck_ratio=bottleneck_ratio,
    ).to(device)
    trainable = [parameter for parameter in expert.parameters() if parameter.requires_grad]
    info = {
        "feature_dim": int(feature_dim),
        "num_prompts": int(num_prompts),
        "bottleneck_dim": feature_dim // bottleneck_ratio,
        "trainable_parameters": trainable,
        "trainable_parameter_count": sum(parameter.numel() for parameter in trainable),
    }
    return expert, info


def build_optimizer(
    expert: MLLMSemanticExpert,
    *,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        [parameter for parameter in expert.parameters() if parameter.requires_grad],
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )


def ensemble_summary(
    prompt_probabilities: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-prompt empirical mean and variance (the prompt ensemble signals)."""

    mean = prompt_probabilities.mean(dim=-1)
    variance = ((prompt_probabilities - mean.unsqueeze(-1)) ** 2).mean(dim=-1)
    return mean, variance
