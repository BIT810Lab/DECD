"""Frozen-feature extraction and the precomputed-feature cache.

Because the backbone is frozen and the consensus gate is deterministic, the
representations ``z0`` are constant throughout adapter training and are
extracted exactly once per prompt set, as stated in the method section.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import DataLoader, TensorDataset

from .backbone import FrozenMLLMBackbone
from .prompts import REASONING_PROMPTS, validate_prompts

FEATURE_FORMAT_VERSION = 1


@torch.inference_mode()
def extract_features(
    backbone: FrozenMLLMBackbone,
    loader: DataLoader,
    *,
    prompts: Sequence[str] = REASONING_PROMPTS,
) -> dict[str, Any]:
    """Run every image through the frozen backbone with each prompt."""

    prompt_tuple = tuple(prompts)
    validate_prompts(prompt_tuple)
    features: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    sample_ids: list[str] = []
    domains: list[str] = []

    for batch in loader:
        encoded = backbone.encode(batch["image"], list(prompt_tuple))
        if encoded.ndim != 3 or encoded.shape[1] != len(prompt_tuple):
            raise ValueError(
                "backbone.encode must return (B, K, d); got "
                f"{tuple(encoded.shape)}"
            )
        features.append(encoded.float().cpu())
        labels.append(batch["label"].float().cpu())
        sample_ids.extend(batch["sample_id"])
        domains.extend(batch["domain"])

    return {
        "features": torch.cat(features, dim=0).contiguous(),
        "labels": torch.cat(labels, dim=0).contiguous(),
        "sample_ids": sample_ids,
        "domains": domains,
        "prompts": list(prompt_tuple),
        "feature_dim": int(features[0].shape[-1]),
    }


def save_feature_cache(path: str | Path, payload: dict[str, Any]) -> None:
    required = {"features", "labels", "sample_ids", "prompts", "feature_dim"}
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"feature cache missing fields: {sorted(missing)}")
    features = payload["features"]
    if features.ndim != 3:
        raise ValueError("cached features must have shape (N, K, d)")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": FEATURE_FORMAT_VERSION,
            **payload,
        },
        output,
    )


def load_feature_cache(path: str | Path) -> dict[str, Any]:
    cache_path = Path(path).expanduser().resolve()
    if not cache_path.is_file():
        raise FileNotFoundError(cache_path)
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "features" not in payload:
        raise ValueError("invalid feature cache")
    payload["features"] = payload["features"].float()
    payload["labels"] = payload["labels"].float()
    return payload


def build_feature_loader(
    cache: dict[str, Any],
    *,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(cache["features"], cache["labels"])
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
