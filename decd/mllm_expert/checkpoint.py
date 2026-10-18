"""Checkpoint handling for the trainable semantic-expert components.

Only the adapter and the linear head are stored: the 7B backbone remains on
its published checkpoint and is never serialized here.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from .model import MLLMSemanticExpert

EXPERT_FORMAT_VERSION = 1


def save_expert_checkpoint(
    path: str | Path,
    *,
    expert: MLLMSemanticExpert,
    config: dict[str, Any],
    prompts: list[str] | tuple[str, ...],
    optimizer: torch.optim.Optimizer | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": EXPERT_FORMAT_VERSION,
        "model_state_dict": expert.state_dict(),
        "optimizer_state_dict": (
            optimizer.state_dict() if optimizer is not None else None
        ),
        "feature_dim": int(expert.feature_dim),
        "num_prompts": int(expert.num_prompts),
        "bottleneck_ratio": int(
            expert.feature_dim // expert.adapter.bottleneck_dim
        ),
        "prompts": list(prompts),
        "config": dict(config),
        "metadata": dict(metadata or {}),
    }
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, output)


def load_expert_checkpoint(
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> tuple[MLLMSemanticExpert, dict[str, Any]]:
    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError("invalid semantic expert checkpoint")

    expert = MLLMSemanticExpert(
        feature_dim=int(payload["feature_dim"]),
        num_prompts=int(payload["num_prompts"]),
        bottleneck_ratio=int(payload.get("bottleneck_ratio", 4)),
    ).to(device)
    expert.load_state_dict(payload["model_state_dict"], strict=True)
    return expert, payload
