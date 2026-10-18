#!/usr/bin/env python3
"""Train the B100 OpenCLIP reference detector."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from decd.visual_expert.checkpoint import (  # noqa: E402
    file_sha256,
    load_yaml_config,
    resolve_project_path,
    save_checkpoint,
    write_json,
    validate_training_config,
)
from decd.visual_expert.data import (  # noqa: E402
    build_eval_loader,
    build_training_loader,
    load_manifest,
)
from decd.visual_expert.losses import detection_loss, gradient_l2_norm, require_finite  # noqa: E402
from decd.visual_expert.manifests import assert_disjoint, content_sha256
from decd.visual_expert.model import (  # noqa: E402
    build_detector,
    build_optimizer,
    collect_predictions,
    set_determinism,
    summarize_predictions,
)

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "visual_expert_reported.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the reported B100 reference detector."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--train-manifest", type=Path)
    parser.add_argument("--dev-manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--max-batches", type=int)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and one balanced data batch without a model.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _path(
    argument: Path | None,
    config_path: Path,
    configured: str,
) -> Path:
    return (
        argument.expanduser().resolve()
        if argument is not None
        else resolve_project_path(config_path, configured)
    )


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    validate_training_config(config)
    train_manifest = _path(
        args.train_manifest,
        args.config,
        config["manifests"]["train"],
    )
    dev_manifest = _path(
        args.dev_manifest,
        args.config,
        config["manifests"]["dev"],
    )
    output = _path(
        args.output,
        args.config,
        config["paths"]["reference_checkpoint"],
    )
    device = torch.device(args.device or config["device"])
    max_batches = int(config['reference_batches'] if args.max_batches is None else args.max_batches)
    seed = int(config["seed"])

    if max_batches <= 0:
        raise ValueError("reference batches must be positive")
    if max_batches > int(config['reference_batches']):
        raise ValueError('--max-batches may shorten B100 for debugging, not extend it')
    if int(config["batch_size"]) % 6 != 0:
        raise ValueError("batch_size must be divisible by 6")
    if device.type == "cuda" and not torch.cuda.is_available() and not args.dry_run:
        raise RuntimeError("CUDA was requested but is unavailable")
    if output.exists() and not args.overwrite and not args.dry_run:
        raise FileExistsError(f"output exists; pass --overwrite: {output}")

    set_determinism(seed)
    train_rows = load_manifest(train_manifest, require_domain=True, role='train')
    dev_rows = load_manifest(dev_manifest, require_domain=True, role='dev')
    assert_disjoint(train_rows, dev_rows)
    train_content_sha256 = content_sha256(train_rows)
    train_loader = build_training_loader(
        train_rows,
        image_size=int(config["image_size"]),
        batch_size=int(config["batch_size"]),
        num_batches=max_batches,
        seed=seed,
        num_workers=int(config["num_workers"]),
        pin_memory=device.type == "cuda",
    )
    dev_loader = build_eval_loader(
        dev_rows,
        image_size=int(config["image_size"]),
        batch_size=int(config["eval_batch_size"]),
        num_workers=int(config["num_workers"]),
        pin_memory=device.type == "cuda",
    )

    if args.dry_run:
        batch = next(iter(train_loader))
        print(
            f"dry-run ok: train={len(train_rows)}, dev={len(dev_rows)}, "
            f"batch_shape={tuple(batch['image'].shape)}"
        )
        return

    detector, parameter_info = build_detector(
        model_name=str(config["model_name"]),
        pretrained=str(config["pretrained"]),
        image_size=int(config["image_size"]),
        trainable_blocks=int(config["trainable_blocks"]),
        device=device,
    )
    optimizer = build_optimizer(
        parameter_info,
        backbone_lr=float(config["backbone_lr"]),
        head_lr=float(config["head_lr"]),
        weight_decay=float(config["weight_decay"]),
    )
    amp_enabled = bool(config["amp"]) and device.type == "cuda"
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    detector.train()
    log_rows: list[dict[str, float | int]] = []
    for batch_index, batch in enumerate(train_loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        require_finite(images, 'training input')
        labels = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            logits, _ = detector(images)
            loss = detection_loss(logits, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradient_l2_norm([p.grad for p in parameter_info['trainable_parameters']])
        scaler.step(optimizer)
        scaler.update()
        for parameter in parameter_info['trainable_parameters']:
            require_finite(parameter, 'updated B100 parameter')

        accuracy = float(
            ((torch.sigmoid(logits) >= config["threshold"]) == labels.bool())
            .float()
            .mean()
            .item()
        )
        log_rows.append(
            {
                "batch": batch_index,
                "detection_loss": float(loss.item()),
                "batch_accuracy": accuracy,
            }
        )
        if batch_index == 1 or batch_index % 10 == 0 or batch_index == max_batches:
            print(
                f"[{batch_index:3d}/{max_batches:3d}] "
                f"loss={loss.item():.6f} accuracy={accuracy:.4f}"
            )

    predictions = collect_predictions(
        detector,
        dev_loader,
        device=device,
        amp_enabled=amp_enabled,
    )
    metrics = summarize_predictions(
        predictions,
        threshold=float(config["threshold"]),
    )
    checkpoint_config = {
        **config,
        "reference_batches": max_batches,
        "device": str(device),
    }
    save_checkpoint(
        output,
        detector=detector,
        optimizer=optimizer,
        config=checkpoint_config,
        feature_dim=int(parameter_info["feature_dim"]),
        metadata={
            "stage": "reference",
            "train_manifest": train_manifest.name,
            "train_manifest_sha256": file_sha256(train_manifest),
            "train_content_sha256": train_content_sha256,
            "completed_batches": max_batches,
            "dev_manifest": dev_manifest.name,
            "dev_manifest_sha256": file_sha256(dev_manifest),
            "train_samples": len(train_rows),
            "dev_samples": len(dev_rows),
            "metrics": metrics,
        },
    )
    write_json(output.with_name("reference_metrics.json"), metrics)
    write_json(output.with_name("reference_train_log.json"), log_rows)
    print(f"saved reference checkpoint: {output}")


if __name__ == "__main__":
    main()
