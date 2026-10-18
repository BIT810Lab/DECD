"""Persistence for the dispatcher MLP, temperatures, and reject threshold."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from .model import ConfidenceAwareDispatcher

DISPATCHER_FORMAT_VERSION = 1


def save_dispatcher_checkpoint(
    path: str | Path,
    *,
    dispatcher: ConfidenceAwareDispatcher,
    config: dict[str, Any],
    temperature_cv: float,
    temperature_mllm: float,
    abstention_threshold_value: float,
    num_prompts: int,
    metrics: dict[str, Any] | None = None,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": DISPATCHER_FORMAT_VERSION,
        "model_state_dict": dispatcher.state_dict(),
        "hidden_dim": int(dispatcher.hidden.out_features),
        "temperature_cv": float(temperature_cv),
        "temperature_mllm": float(temperature_mllm),
        "abstention_threshold": float(abstention_threshold_value),
        "num_prompts": int(num_prompts),
        "config": dict(config),
        "metrics": dict(metrics or {}),
    }
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, output)


def load_dispatcher_checkpoint(
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> tuple[ConfidenceAwareDispatcher, dict[str, Any]]:
    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError("invalid dispatcher checkpoint")
    dispatcher = ConfidenceAwareDispatcher(
        hidden_dim=int(payload.get("hidden_dim", 64))
    ).to(device)
    dispatcher.load_state_dict(payload["model_state_dict"], strict=True)
    dispatcher.eval()
    return dispatcher, payload
