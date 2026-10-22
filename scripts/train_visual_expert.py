#!/usr/bin/env python3
"""Continue B100 reference training with reported real-only risk repair."""

# SHA256 rejects mismatched reference, Train inputs and risk profiles during
# repair continuation. It is training validation, not dataset construction.

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
    restore_detector,
    save_checkpoint,
    write_json,
    validate_training_config,
)
from decd.visual_expert.data import (  # noqa: E402
    build_eval_loader,
    build_training_loader,
    load_manifest,
)
from decd.visual_expert.losses import (  # noqa: E402
    adaptive_auxiliary_scale,
    assign_combined_gradients,
    compute_gradients,
    detection_loss,
    linear_ramp,
    real_only_margin_repair,
    require_finite,
)
from decd.visual_expert.model import (  # noqa: E402
    collect_predictions,
    set_determinism,
    summarize_predictions,
)
from decd.visual_expert.risk import (  # noqa: E402
    load_risk_profile,
    validate_risk_profile,
    weight_tensor,
)
from decd.visual_expert.manifests import assert_disjoint, content_sha256

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "visual_expert_reported.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the reported B50 Visual Expert with real-only selective "
            "margin repair."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--train-manifest", type=Path)
    parser.add_argument("--dev-manifest", type=Path)
    parser.add_argument("--reference-checkpoint", type=Path)
    parser.add_argument("--risk-profile", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--max-batches", type=int)
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
    reference_checkpoint = _path(
        args.reference_checkpoint,
        args.config,
        config["paths"]["reference_checkpoint"],
    )
    risk_profile_path = _path(
        args.risk_profile,
        args.config,
        config["paths"]["risk_profile"],
    )
    output = _path(
        args.output,
        args.config,
        config["paths"]["visual_expert_checkpoint"],
    )
    device = torch.device(args.device or config["device"])
    max_batches = int(config['repair_batches'] if args.max_batches is None else args.max_batches)

    if str(config["repair_label_side"]).lower() != "real":
        raise ValueError("the published Visual Expert supports real-only repair")
    if max_batches <= 0:
        raise ValueError("repair batches must be positive")
    if max_batches > int(config['repair_batches']):
        raise ValueError('--max-batches may shorten B50 for debugging, not extend it')
    if int(config["batch_size"]) % 6 != 0:
        raise ValueError("batch_size must be divisible by 6")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    protected_outputs = (output, output.with_name('visual_expert_metrics.json'),
                         output.with_name('visual_expert_train_log.json'))
    if not args.overwrite and any(p.exists() for p in protected_outputs):
        raise FileExistsError('repair output/checkpoint/log exists; choose a new directory or --overwrite')

    set_determinism(int(config["seed"]))
    profile = load_risk_profile(risk_profile_path)
    train_rows = load_manifest(train_manifest, require_domain=True, role='train')
    dev_rows = load_manifest(dev_manifest, require_domain=True, role='dev')
    assert_disjoint(train_rows, dev_rows)
    train_content_sha256 = content_sha256(train_rows)
    if profile.get('provenance', {}).get('train_content_sha256') != train_content_sha256:
        raise ValueError('risk profile content identity differs from active Train')

    detector, optimizer, parameter_info, reference_metadata = restore_detector(
        reference_checkpoint,
        device=device,
        restore_optimizer=True,
        expected_config=config,
        require_reference=True,
        train_manifest_sha256=file_sha256(train_manifest),
        train_content_sha256=train_content_sha256,
    )
    if optimizer is None:
        raise RuntimeError("optimizer restoration failed")

    train_manifest_sha256 = file_sha256(train_manifest)
    validate_risk_profile(
        profile,
        risk_column=str(config["risk_column"]),
        gamma=float(config["gamma"]),
        normalization=str(config["risk_normalization"]),
        normalization_alpha=float(config["risk_normalization_alpha"]),
        train_manifest_sha256=train_manifest_sha256,
        reference_checkpoint_sha256=reference_metadata[
            "checkpoint_sha256"
        ],
    )

    train_loader = build_training_loader(
        train_rows,
        image_size=int(reference_metadata["image_size"]),
        batch_size=int(config["batch_size"]),
        num_batches=max_batches,
        seed=int(config["seed"]),
        num_workers=int(config["num_workers"]),
        pin_memory=device.type == "cuda",
    )
    dev_loader = build_eval_loader(
        dev_rows,
        image_size=int(reference_metadata["image_size"]),
        batch_size=int(config["eval_batch_size"]),
        num_workers=int(config["num_workers"]),
        pin_memory=device.type == "cuda",
    )
    weights = weight_tensor(profile, device=device)
    trainable_parameters = parameter_info["trainable_parameters"]
    amp_enabled = bool(config["amp"]) and device.type == "cuda"

    detector.train()
    log_rows: list[dict[str, float | int]] = []
    for batch_index, batch in enumerate(train_loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        require_finite(images, 'training input')
        labels = batch["label"].to(device, non_blocking=True)
        domain_ids = batch["domain_id"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            logits_half, _ = detector(images)
        logits = logits_half.float()
        loss_detection = detection_loss(logits, labels)
        loss_repair, repair_diagnostics = real_only_margin_repair(
            logits=logits,
            labels=labels,
            domain_ids=domain_ids,
            weight_matrix=weights,
            gamma=float(config["gamma"]),
        )

        detection_gradients = compute_gradients(
            loss_detection,
            trainable_parameters,
            retain_graph=True,
        )
        repair_gradients = compute_gradients(
            loss_repair,
            trainable_parameters,
            retain_graph=False,
        )
        rho, gradient_diagnostics = adaptive_auxiliary_scale(
            detection_gradients,
            repair_gradients,
            rho_max=float(config["rho_max"]),
            epsilon=float(config["gradient_epsilon"]),
        )
        lambda_now = float(config["repair_lambda"]) * linear_ramp(
            batch_index,
            int(config["ramp_batches"]),
        )
        final_norm = assign_combined_gradients(
            parameters=trainable_parameters,
            detection_gradients=detection_gradients,
            repair_gradients=repair_gradients,
            repair_lambda=lambda_now,
            rho=rho,
        )
        optimizer.step()
        for parameter in trainable_parameters:
            require_finite(parameter, 'updated B50 parameter')

        with torch.no_grad():
            accuracy = float(
                (
                    (torch.sigmoid(logits) >= float(config["threshold"]))
                    == labels.bool()
                )
                .float()
                .mean()
                .item()
            )
        row = {
            "batch": batch_index,
            "detection_loss": float(loss_detection.item()),
            "repair_loss": float(loss_repair.item()),
            "batch_accuracy": accuracy,
            "repair_lambda": lambda_now,
            "rho": rho,
            "detection_gradient_norm": gradient_diagnostics[
                "detection_gradient_norm"
            ],
            "repair_gradient_norm": gradient_diagnostics[
                "repair_gradient_norm"
            ],
            "final_gradient_norm": final_norm,
            "repair_active_rate": float(
                repair_diagnostics["active_rate"].item()
            ),
        }
        log_rows.append(row)
        if batch_index == 1 or batch_index % 10 == 0 or batch_index == max_batches:
            print(
                f"[{batch_index:3d}/{max_batches:3d}] "
                f"det={row['detection_loss']:.6f} "
                f"repair={row['repair_loss']:.6f} "
                f"lambda={lambda_now:.4f} rho={rho:.4f}"
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
        "repair_batches": max_batches,
        "device": str(device),
    }
    save_checkpoint(
        output,
        detector=detector,
        optimizer=optimizer,
        config=checkpoint_config,
        feature_dim=int(reference_metadata["feature_dim"]),
        metadata={
            "stage": "visual_expert",
            "reference_checkpoint": reference_checkpoint.name,
            "reference_checkpoint_sha256": reference_metadata[
                "checkpoint_sha256"
            ],
            "risk_profile": risk_profile_path.name,
            "risk_profile_sha256": file_sha256(risk_profile_path),
            "train_manifest": train_manifest.name,
            "train_manifest_sha256": train_manifest_sha256,
            "train_content_sha256": train_content_sha256,
            "completed_batches": max_batches,
            "dev_manifest": dev_manifest.name,
            "dev_manifest_sha256": file_sha256(dev_manifest),
            "optimizer_restored": True,
            "metrics": metrics,
        },
    )
    write_json(output.with_name("visual_expert_metrics.json"), metrics)
    write_json(output.with_name("visual_expert_train_log.json"), log_rows)
    print(f"saved Visual Expert checkpoint: {output}")


if __name__ == "__main__":
    main()
