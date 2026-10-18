"""Tests for MLLM manifests, image loading, feature extraction, and cache."""

import csv

import numpy as np
import pytest
import torch
from PIL import Image

from decd.mllm_expert.data import (
    build_image_loader,
    load_mllm_manifest,
)
from decd.mllm_expert.features import (
    build_feature_loader,
    extract_features,
    load_feature_cache,
    save_feature_cache,
)
from decd.mllm_expert.prompts import REASONING_PROMPTS


class DummyBackbone:
    """Stand-in for the frozen Qwen backbone; no transformers dependency."""

    feature_dim = 16

    def encode(self, images, prompts):
        batch = len(images)
        num_prompts = len(prompts)
        values = torch.arange(
            batch * num_prompts * self.feature_dim,
            dtype=torch.float32,
        ).reshape(batch, num_prompts, self.feature_dim)
        return (values.remainder(1000) / 1000.0)


def _write_image(path, seed):
    rng = np.random.default_rng(seed)
    array = rng.integers(0, 255, (24, 24, 3), dtype=np.uint8)
    Image.fromarray(array).save(path)


def _write_manifest(path, image_paths):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_id", "image_path", "domain", "label"],
        )
        writer.writeheader()
        for index, image_path in enumerate(image_paths):
            writer.writerow(
                {
                    "sample_id": f"id_{index}",
                    "image_path": image_path.name,
                    "domain": "Cartoon",
                    "label": index % 2,
                }
            )


def test_manifest_image_loader_and_extraction(tmp_path):
    image_paths = []
    for index in range(4):
        path = tmp_path / f"img_{index}.png"
        _write_image(path, index)
        image_paths.append(path)
    manifest_path = tmp_path / "manifest.csv"
    _write_manifest(manifest_path, image_paths)

    rows = load_mllm_manifest(manifest_path)
    assert [row["sample_id"] for row in rows] == [f"id_{i}" for i in range(4)]
    assert [row["label"] for row in rows] == [0, 1, 0, 1]
    assert rows[0]["image_path"] == image_paths[0].resolve()

    loader = build_image_loader(rows, batch_size=2, shuffle=False)
    batch = next(iter(loader))
    assert len(batch["image"]) == 2
    assert batch["image"][0].mode == "RGB"
    assert batch["label"].shape == (2,)

    payload = extract_features(
        DummyBackbone(),
        loader,
        prompts=REASONING_PROMPTS,
    )
    assert payload["features"].shape == (4, 3, 16)
    assert payload["labels"].shape == (4,)
    assert payload["prompts"] == list(REASONING_PROMPTS)
    assert payload["sample_ids"] == [f"id_{i}" for i in range(4)]
    assert torch.all((payload["features"] >= 0) & (payload["features"] < 1))


def test_feature_cache_roundtrip_and_loader(tmp_path):
    payload = {
        "features": torch.randn(10, 3, 16),
        "labels": torch.randint(0, 2, (10,)).float(),
        "sample_ids": [f"s_{i}" for i in range(10)],
        "domains": ["all"] * 10,
        "prompts": list(REASONING_PROMPTS),
        "feature_dim": 16,
    }
    cache_path = tmp_path / "features.pt"
    save_feature_cache(cache_path, payload)
    loaded = load_feature_cache(cache_path)
    assert torch.allclose(loaded["features"], payload["features"])
    assert len(loaded["prompts"]) == 3
    assert loaded["feature_dim"] == 16

    loader = build_feature_loader(loaded, batch_size=4, shuffle=False)
    features, labels = next(iter(loader))
    assert features.shape == (4, 3, 16)
    assert labels.shape == (4,)


def test_missing_image_raises(tmp_path):
    manifest_path = tmp_path / "manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as handle:
        handle.write("image_path,label\n")
        handle.write("does_not_exist.png,1\n")
    with pytest.raises(FileNotFoundError):
        load_mllm_manifest(manifest_path)


def test_importing_backbone_does_not_require_transformers(monkeypatch):
    # The lazy import keeps configuration and tests independent of a 7B model.
    # A cached entry of None makes `import transformers` raise ImportError,
    # simulating an environment without transformers installed without
    # uninstalling the real package (which would trigger a real model
    # download on environments that do have it).
    import importlib
    import sys

    import decd.mllm_expert.backbone as backbone_module

    monkeypatch.setitem(sys.modules, "transformers", None)
    importlib.reload(backbone_module)
    try:
        with pytest.raises(RuntimeError, match="transformers"):
            backbone_module.Qwen25VLBackbone(
                "Qwen/Qwen2.5-VL-7B-Instruct",
                device="cpu",
                dtype=torch.float32,
            )
    finally:
        # Restore the module for later tests even if the assertion fails.
        monkeypatch.undo()
        importlib.reload(backbone_module)
