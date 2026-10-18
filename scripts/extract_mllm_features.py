#!/usr/bin/env python3
"""Extract frozen Qwen2.5-VL prompt-conditioned features for one split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from decd.mllm_expert.backbone import Qwen25VLBackbone  # noqa: E402
from decd.mllm_expert.data import build_image_loader, load_mllm_manifest  # noqa: E402
from decd.mllm_expert.features import (  # noqa: E402
    extract_features,
    save_feature_cache,
)
from decd.mllm_expert.prompts import REASONING_PROMPTS  # noqa: E402

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "mllm_expert_reported.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache frozen MLLM features z0 for a manifest split."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--model-name")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    manifest = args.manifest.expanduser().resolve()
    output = args.output.expanduser().resolve()
    device = torch.device(args.device or config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"output exists; pass --overwrite: {output}")

    rows = load_mllm_manifest(manifest)
    loader = build_image_loader(
        rows,
        batch_size=int(args.batch_size or config["feature_batch_size"]),
        shuffle=False,
        num_workers=int(args.num_workers),
    )
    backbone = Qwen25VLBackbone(
        args.model_name or config["backbone_name"],
        device=device,
        # bfloat16 avoids fp16 overflows (NaN pooled features) observed in
        # the Qwen2.5-VL vision tower; see backbone.py for details.
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        pooling=str(config.get("pooling", "mean")),
    )
    payload = extract_features(backbone, loader, prompts=REASONING_PROMPTS)
    if payload["feature_dim"] != int(config["feature_dim"]):
        raise ValueError(
            f"backbone feature dim {payload['feature_dim']} does not match "
            f"configured {config['feature_dim']}"
        )
    save_feature_cache(output, {**payload, "manifest": manifest.name})
    print(
        f"cached {payload['features'].shape[0]} samples x "
        f"{len(REASONING_PROMPTS)} prompts x {payload['feature_dim']}d -> {output}"
    )


if __name__ == "__main__":
    main()
