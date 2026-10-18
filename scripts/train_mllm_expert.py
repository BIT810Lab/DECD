#!/usr/bin/env python3
"""Train the PP adapter and head on cached frozen MLLM features."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from decd.mllm_expert.checkpoint import save_expert_checkpoint  # noqa: E402
from decd.mllm_expert.features import (  # noqa: E402
    build_feature_loader,
    load_feature_cache,
)
from decd.mllm_expert.model import (  # noqa: E402
    build_optimizer,
    build_semantic_expert,
)
from decd.mllm_expert.prompts import REASONING_PROMPTS  # noqa: E402
from decd.mllm_expert.training import (  # noqa: E402
    evaluate_expert,
    set_determinism,
    train_semantic_expert,
)

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "mllm_expert_reported.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the prior-preserved MLLM semantic expert."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--train-features", type=Path)
    parser.add_argument("--dev-features", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _path(argument: Path | None, config_path: Path, configured: str) -> Path:
    if argument is not None:
        return argument.expanduser().resolve()
    return (config_path.parent.parent / configured).resolve()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    train_cache_path = _path(
        args.train_features,
        args.config,
        config["features"]["train"],
    )
    dev_cache_path = _path(
        args.dev_features,
        args.config,
        config["features"]["dev"],
    )
    output = _path(
        args.output,
        args.config,
        config["paths"]["mllm_expert_checkpoint"],
    )
    device = torch.device(args.device or config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"output exists; pass --overwrite: {output}")

    set_determinism(int(config["seed"]))
    train_cache = load_feature_cache(train_cache_path)
    dev_cache = load_feature_cache(dev_cache_path)
    if train_cache["prompts"] != dev_cache["prompts"]:
        raise ValueError("train and dev caches were built with different prompts")
    if train_cache["prompts"] != list(REASONING_PROMPTS):
        raise ValueError("feature-cache prompts do not match REASONING_PROMPTS")
    if train_cache["feature_dim"] != int(config["feature_dim"]):
        raise ValueError("feature cache dim does not match configuration")

    expert, info = build_semantic_expert(
        feature_dim=int(config["feature_dim"]),
        num_prompts=int(config["num_prompts"]),
        bottleneck_ratio=int(config["bottleneck_ratio"]),
        device=device,
        prompts=REASONING_PROMPTS,
    )
    optimizer = build_optimizer(
        expert,
        learning_rate=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    train_loader = build_feature_loader(
        train_cache,
        batch_size=int(config["batch_size"]),
        shuffle=True,
    )
    dev_loader = build_feature_loader(
        dev_cache,
        batch_size=int(config.get("eval_batch_size", 32)),
        shuffle=False,
    )

    result = train_semantic_expert(
        expert,
        train_loader,
        dev_loader,
        optimizer=optimizer,
        epochs=int(args.epochs or config["epochs"]),
        np_temperature=float(config["np_temperature"]),
        np_lambda=float(config["np_lambda"]),
        device=device,
        selection_metric=str(config["selection_metric"]),
        early_stopping_patience=(
            int(config["early_stopping_patience"])
            if config.get("early_stopping_patience")
            else None
        ),
    )
    final_dev = evaluate_expert(
        expert,
        dev_loader,
        device=device,
        threshold=float(config["threshold"]),
    )
    save_expert_checkpoint(
        output,
        expert=expert,
        config=config,
        prompts=list(REASONING_PROMPTS),
        optimizer=optimizer,
        metadata={
            "train_features": train_cache_path.name,
            "dev_features": dev_cache_path.name,
            "trainable_parameter_count": info["trainable_parameter_count"],
            "best_epoch": result["best_epoch"],
            "dev": {
                key: final_dev[key]
                for key in ("accuracy", "macro_f1", "mean_prompt_variance", "n")
            },
        },
    )
    _write_json(output.with_name("mllm_expert_metrics.json"), {
        "dev": {
            key: final_dev[key]
            for key in ("accuracy", "macro_f1", "mean_prompt_variance", "n")
        },
        "best_epoch": result["best_epoch"],
        "history": result["history"],
        "trainable_parameter_count": info["trainable_parameter_count"],
    })
    print(f"saved MLLM expert checkpoint: {output}")


if __name__ == "__main__":
    main()
