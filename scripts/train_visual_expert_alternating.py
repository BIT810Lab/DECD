#!/usr/bin/env python3
"""Train the full-data MOSAIC v2 Visual Expert with B100/B50 alternation.

Protocol:
  1. 700 BCE-only reference updates (one full balanced pass over Train).
  2. Compute a fixed risk profile from all Train predictions under the reference.
  3. Continue the same model and AdamW optimizer for 14 cycles of 100 BCE-only
     updates followed by 50 joint detection + real-side repair updates.

The BCE and repair DataLoader iterators are each created once and consumed
across cycles. Their sampler queues therefore advance continuously.
"""

# SHA256 validates published split bytes, initialization and checkpoint/profile
# provenance for training and resume. It is not dataset construction.

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from decd.visual_expert.checkpoint import (  # noqa: E402
    file_sha256,
    load_checkpoint,
    load_yaml_config,
    resolve_project_path,
    restore_detector,
    save_checkpoint,
    write_json,
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
)
from decd.visual_expert.model import (  # noqa: E402
    build_detector,
    build_optimizer,
    collect_predictions,
    set_determinism,
    summarize_predictions,
)
from decd.visual_expert.protocol import (  # noqa: E402
    validate_balanced_split,
    validate_exact_exposures,
    validate_train_dev_integrity,
    validate_training_schedule,
    validate_resume_contract,
)
from decd.visual_expert.risk import (  # noqa: E402
    build_risk_profile,
    load_risk_profile,
    save_risk_profile,
    validate_risk_profile,
    weight_tensor,
)
from decd.visual_expert.sampler import DomainLabelBalancedBatchSampler  # noqa: E402

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "visual_expert_mosaic_v2_full.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the full-data MOSAIC v2 Visual Expert."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate manifests, image paths, split sizes, and sampling coverage without training.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest reference/cycle checkpoint in the configured run directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Start from scratch and overwrite configured outputs. Do not combine with --resume.",
    )
    return parser.parse_args()


def _path(config_path: Path, configured: str) -> Path:
    return resolve_project_path(config_path, configured)


def _amp_context(device: torch.device, enabled: bool):
    return torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=enabled,
    )


def _new_scaler(amp_enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=amp_enabled)
    return torch.cuda.amp.GradScaler(enabled=amp_enabled)


def _record_batch_exposures(batch: dict[str, Any], counts: Counter[str]) -> None:
    counts.update(str(sample_id) for sample_id in batch["sample_id"])


def _bce_update(
    *,
    detector: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    batch: dict[str, Any],
    device: torch.device,
    amp_enabled: bool,
    threshold: float,
) -> dict[str, float]:
    detector.train()
    images = batch["image"].to(device, non_blocking=True)
    labels = batch["label"].to(device, non_blocking=True)
    optimizer.zero_grad(set_to_none=True)
    with _amp_context(device, amp_enabled):
        logits, _ = detector(images)
        loss = detection_loss(logits.float(), labels)
    if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(logits).all()):
        raise FloatingPointError("Non-finite BCE loss/logits")
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                raise FloatingPointError("Non-finite BCE gradient; update aborted")
    scaler.step(optimizer)
    scaler.update()
    with torch.no_grad():
        accuracy = float(
            ((torch.sigmoid(logits.float()) >= threshold) == labels.bool())
            .float().mean().item()
        )
    return {"detection_loss": float(loss.item()), "batch_accuracy": accuracy}


def _joint_update(
    *,
    detector: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    trainable_parameters: list[torch.nn.Parameter],
    batch: dict[str, Any],
    weights: torch.Tensor,
    config: dict[str, Any],
    device: torch.device,
    amp_enabled: bool,
    repair_update: int,
) -> dict[str, float]:
    detector.train()
    images = batch["image"].to(device, non_blocking=True)
    labels = batch["label"].to(device, non_blocking=True)
    domain_ids = batch["domain_id"].to(device, non_blocking=True)
    optimizer.zero_grad(set_to_none=True)
    with _amp_context(device, amp_enabled):
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
    if not all(bool(torch.isfinite(x).all()) for x in (logits, loss_detection, loss_repair)):
        raise FloatingPointError("Non-finite repair loss/logits")
    detection_gradients = compute_gradients(
        loss_detection, trainable_parameters, retain_graph=True
    )
    repair_gradients = compute_gradients(
        loss_repair, trainable_parameters, retain_graph=False
    )
    if any(g is not None and not bool(torch.isfinite(g).all()) for g in detection_gradients + repair_gradients):
        raise FloatingPointError("Non-finite repair gradient; update aborted")
    rho, gradient_diagnostics = adaptive_auxiliary_scale(
        detection_gradients,
        repair_gradients,
        rho_max=float(config["rho_max"]),
        epsilon=float(config["gradient_epsilon"]),
    )
    repair_lambda = float(config["repair_lambda"]) * linear_ramp(
        repair_update, int(config["ramp_batches"])
    )
    final_norm = assign_combined_gradients(
        parameters=trainable_parameters,
        detection_gradients=detection_gradients,
        repair_gradients=repair_gradients,
        repair_lambda=repair_lambda,
        rho=rho,
    )
    if not math.isfinite(final_norm) or any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in trainable_parameters):
        raise FloatingPointError("Non-finite combined gradient; update aborted")
    optimizer.step()
    with torch.no_grad():
        accuracy = float(
            ((torch.sigmoid(logits) >= float(config["threshold"])) == labels.bool())
            .float().mean().item()
        )
    return {
        "detection_loss": float(loss_detection.item()),
        "repair_loss": float(loss_repair.item()),
        "batch_accuracy": accuracy,
        "repair_lambda": repair_lambda,
        "rho": rho,
        "detection_gradient_norm": gradient_diagnostics["detection_gradient_norm"],
        "repair_gradient_norm": gradient_diagnostics["repair_gradient_norm"],
        "final_gradient_norm": final_norm,
        "repair_active_rate": float(repair_diagnostics["active_rate"].item()),
    }


def _planned_coverage(
    rows: list[dict[str, Any]], *, batches: int, seed: int, expected_repeats: int, phase: str
) -> dict[str, int]:
    sampler = DomainLabelBalancedBatchSampler(
        rows, batch_size=6, num_batches=batches, seed=seed
    )
    counts: Counter[str] = Counter()
    for batch_indices in sampler:
        counts.update(str(rows[index]["sample_id"]) for index in batch_indices)
    return validate_exact_exposures(
        rows, counts, expected_repeats=expected_repeats, phase_name=phase
    )


def _build_risk_profile(
    *,
    detector: torch.nn.Module,
    train_rows: list[dict[str, Any]],
    train_eval_loader: Any,
    config: dict[str, Any],
    device: torch.device,
    amp_enabled: bool,
    train_manifest: Path,
    reference_checkpoint: Path,
    risk_profile_path: Path,
) -> dict[str, Any]:
    predictions = collect_predictions(
        detector, train_eval_loader, device=device, amp_enabled=amp_enabled
    )
    if len(predictions) != len(train_rows):
        raise RuntimeError(
            f"risk profiling used {len(predictions)} predictions for "
            f"{len(train_rows)} Train rows"
        )
    if [str(p["sample_id"]) for p in predictions] != [str(r["sample_id"]) for r in train_rows]:
        raise RuntimeError("Risk profiling prediction identity/order mismatch")
    profile = build_risk_profile(
        predictions,
        gamma=float(config["gamma"]),
        risk_column=str(config["risk_column"]),
        normalization=str(config["risk_normalization"]),
        alpha=float(config["risk_normalization_alpha"]),
        provenance={
            "manifest": train_manifest.name,
            "manifest_sha256": file_sha256(train_manifest),
            "reference_checkpoint": reference_checkpoint.name,
            "reference_checkpoint_sha256": file_sha256(reference_checkpoint),
            "n_train_predictions": len(predictions),
        },
    )
    save_risk_profile(risk_profile_path, profile)
    return profile


def _validate_profile(
    *, profile: dict[str, Any], config: dict[str, Any], train_manifest: Path, reference_checkpoint: Path
) -> None:
    validate_risk_profile(
        profile,
        risk_column=str(config["risk_column"]),
        gamma=float(config["gamma"]),
        normalization=str(config["risk_normalization"]),
        normalization_alpha=float(config["risk_normalization_alpha"]),
        train_manifest_sha256=file_sha256(train_manifest),
        reference_checkpoint_sha256=file_sha256(reference_checkpoint),
    )


def main() -> None:
    args = parse_args()
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    config = load_yaml_config(args.config)
    validate_training_schedule(config)
    device = torch.device("cpu" if args.dry_run else (args.device or config["device"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if int(config["batch_size"]) != 6:
        raise ValueError("this protocol requires batch_size=6 (one per domain-label cell)")
    if str(config["repair_label_side"]).lower() != "real":
        raise ValueError("this training entrypoint implements real-side repair only")

    train_manifest = _path(args.config, config["manifests"]["train"])
    dev_manifest = _path(args.config, config["manifests"]["dev"])
    reference_checkpoint = _path(args.config, config["paths"]["reference_checkpoint"])
    risk_profile_path = _path(args.config, config["paths"]["risk_profile"])
    final_checkpoint = _path(args.config, config["paths"]["visual_expert_checkpoint"])
    evaluation_dir = _path(args.config, config["paths"]["evaluation_dir"])
    training_log_path = _path(args.config, config["paths"]["training_log"])
    run_dir = _path(args.config, config["paths"]["run_dir"])
    summary_path = run_dir / "training_summary.json"

    required_outputs = [reference_checkpoint, risk_profile_path, final_checkpoint, training_log_path]
    existing = [str(path) for path in required_outputs if path.exists()]
    if existing and not (args.resume or args.overwrite or args.dry_run):
        raise FileExistsError(
            "refusing to overwrite existing experiment outputs; use a new run_dir, "
            "--resume, or --overwrite explicitly: " + ", ".join(existing)
        )

    if not args.dry_run:
        set_determinism(int(config["seed"]))
    # Optional SHA256 pins verify published split CSVs for this training run.
    # No construction report or private candidate-pool index is required.
    manifest_pins = config.get("manifest_sha256", {})
    for role, manifest in (("train", train_manifest), ("dev", dev_manifest)):
        if role in manifest_pins and file_sha256(manifest) != manifest_pins[role]:
            raise ValueError(f"Published {role} manifest SHA256 mismatch")
    # Check manifest cardinalities before checking every image path, so an old
    # or partial manifest fails with a useful split-size error first.
    train_rows_manifest = load_manifest(
        train_manifest, require_domain=True, check_files=False, role="train"
    )
    dev_rows_manifest = load_manifest(
        dev_manifest, require_domain=True, check_files=False, role="dev"
    )
    train_counts = validate_balanced_split(
        train_rows_manifest,
        expected_per_cell=int(config["expected_train_per_cell"]),
        split_name="Train",
    )
    dev_counts = validate_balanced_split(
        dev_rows_manifest,
        expected_per_cell=int(config["expected_dev_per_cell"]),
        split_name="Dev",
    )
    # Now verify that every image referenced by the validated manifests exists.
    train_rows = load_manifest(train_manifest, require_domain=True, check_files=True, role="train")
    dev_rows = load_manifest(dev_manifest, require_domain=True, check_files=True, role="dev")
    integrity = validate_train_dev_integrity(train_rows, dev_rows)
    print(f"Train validated: {len(train_rows)} samples; cells={train_counts}")
    print(f"Dev validated: {len(dev_rows)} samples; cells={dev_counts}")
    print(f"Train/Dev bytes and available groups verified: {integrity}", flush=True)
    # Three checkpoint-sized allocations cover reference + final + atomic-save
    # temporary file. No output directory is created by this CPU preflight.
    free_bytes = shutil.disk_usage(PROJECT_ROOT).free
    disk_required = int((3 * float(config.get("checkpoint_estimate_gib", 2.0)) + float(config.get("minimum_free_disk_gib", 1.0))) * 1024**3)
    if free_bytes < disk_required:
        raise OSError(f"Disk safety preflight failed: free={free_bytes}, required={disk_required}")
    print(f"Disk preflight: free={free_bytes / 1024**3:.2f} GiB; required={disk_required / 1024**3:.2f} GiB", flush=True)
    initial_sha = config.get("initialization_sha256")
    if initial_sha and file_sha256(Path(str(config["pretrained"]))) != initial_sha:
        raise ValueError("OpenCLIP initialization file SHA mismatch")

    reference_batches = int(config["reference_batches"])
    cycles = int(config["cycles"])
    bce_per_cycle = int(config["bce_batches_per_cycle"])
    repair_per_cycle = int(config["repair_batches_per_cycle"])
    total_bce_continuation = cycles * bce_per_cycle
    total_repair = cycles * repair_per_cycle
    expected_per_cell = int(config["expected_train_per_cell"])
    if reference_batches != expected_per_cell:
        raise ValueError("reference_batches must equal expected_train_per_cell for one full pass")
    if total_repair != expected_per_cell:
        raise ValueError("cycles * repair_batches_per_cycle must equal expected_train_per_cell")
    if total_bce_continuation % expected_per_cell != 0:
        raise ValueError("total BCE continuation updates must be a whole number of balanced passes")

    coverage_plan = {
        "reference": _planned_coverage(
            train_rows, batches=reference_batches, seed=int(config["seed"]),
            expected_repeats=1, phase="reference sampler plan"
        ),
        "bce_continuation": _planned_coverage(
            train_rows, batches=total_bce_continuation, seed=int(config["bce_sampler_seed"]),
            expected_repeats=total_bce_continuation // expected_per_cell,
            phase="BCE continuation sampler plan"
        ),
        "joint_repair": _planned_coverage(
            train_rows, batches=total_repair, seed=int(config["repair_sampler_seed"]),
            expected_repeats=1, phase="joint repair sampler plan"
        ),
    }

    reference_loader = build_training_loader(
        train_rows,
        image_size=int(config["image_size"]),
        batch_size=6,
        num_batches=reference_batches,
        seed=int(config["seed"]),
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
        first_batch = next(iter(reference_loader))
        print(
            "dry-run passed: "
            f"batch_shape={tuple(first_batch['image'].shape)}, "
            f"reference_updates={reference_batches}, cycles={cycles}, "
            f"BCE_updates={total_bce_continuation}, joint_updates={total_repair}, "
            f"coverage={coverage_plan}"
        )
        return

    amp_enabled = bool(config["amp"]) and device.type == "cuda"
    scaler = _new_scaler(amp_enabled)
    logs: list[dict[str, Any]] = []
    reference_dev_metrics: dict[str, Any] | None = None
    completed_cycle = 0
    resume_from_reference = False
    profile: dict[str, Any] | None = None

    # Resume either from the latest completed cycle or from the saved reference
    # checkpoint if interruption occurred before the first alternating cycle.
    if args.resume:
        if final_checkpoint.is_file():
            resume_payload = load_checkpoint(final_checkpoint)
            completed_cycle = validate_resume_contract(
                resume_payload, config, train_sha256=file_sha256(train_manifest),
                dev_sha256=file_sha256(dev_manifest),
                reference_sha256=file_sha256(reference_checkpoint),
                risk_profile_sha256=file_sha256(risk_profile_path),
            )
            detector, optimizer, parameter_info, _ = restore_detector(
                final_checkpoint, device=device, restore_optimizer=True
            )
            stored_metadata = resume_payload.get("metadata", {})
        elif reference_checkpoint.is_file():
            resume_payload = load_checkpoint(reference_checkpoint)
            validate_resume_contract(resume_payload, config,
                train_sha256=file_sha256(train_manifest), dev_sha256=file_sha256(dev_manifest))
            detector, optimizer, parameter_info, _ = restore_detector(
                reference_checkpoint, device=device, restore_optimizer=True
            )
            stored_metadata = resume_payload.get("metadata", {})
            completed_cycle = 0
            resume_from_reference = True
        else:
            raise FileNotFoundError(
                "cannot resume: neither a completed-cycle checkpoint nor a reference checkpoint exists"
            )
        if optimizer is None:
            raise RuntimeError("resume checkpoint does not contain AdamW state")
        scaler_state = stored_metadata.get("grad_scaler_state_dict")
        if scaler_state is None:
            raise RuntimeError("resume checkpoint is missing GradScaler state")
        scaler.load_state_dict(scaler_state)
        if not training_log_path.is_file():
            logs = []
        else:
            logs = json.loads(training_log_path.read_text(encoding="utf-8"))
            logs = [entry for entry in logs if int(entry.get("cycle", 0)) <= completed_cycle]
        if not reference_checkpoint.is_file():
            raise FileNotFoundError("resume requires the reference checkpoint")
        if risk_profile_path.is_file():
            profile = load_risk_profile(risk_profile_path)
            _validate_profile(
                profile=profile, config=config, train_manifest=train_manifest,
                reference_checkpoint=reference_checkpoint
            )
        elif not resume_from_reference:
            raise FileNotFoundError(
                "risk profile is missing; a cycle checkpoint cannot rebuild it "
                "because its weights are no longer the reference model"
            )
        if (evaluation_dir / "reference_dev_metrics.json").is_file():
            reference_dev_metrics = json.loads(
                (evaluation_dir / "reference_dev_metrics.json").read_text(encoding="utf-8")
            )
        print(f"Resuming from completed_cycle={completed_cycle}")
    else:
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
        reference_exposures: Counter[str] = Counter()
        detector.train()
        for update, batch in enumerate(reference_loader, start=1):
            _record_batch_exposures(batch, reference_exposures)
            stats = _bce_update(
                detector=detector, optimizer=optimizer, scaler=scaler, batch=batch,
                device=device, amp_enabled=amp_enabled, threshold=float(config["threshold"])
            )
            logs.append({"phase": "reference", "update": update, **stats})
            if update == 1 or update % 50 == 0 or update == reference_batches:
                print(
                    f"[reference {update}/{reference_batches}] "
                    f"det={stats['detection_loss']:.6f} acc={stats['batch_accuracy']:.4f}"
                )
        reference_coverage = validate_exact_exposures(
            train_rows, reference_exposures, expected_repeats=1, phase_name="reference"
        )
        save_checkpoint(
            reference_checkpoint,
            detector=detector,
            optimizer=optimizer,
            config={**config, "device": str(device)},
            feature_dim=int(parameter_info["feature_dim"]),
            metadata={
                "stage": "reference",
                "train_manifest": train_manifest.name,
                "train_manifest_sha256": file_sha256(train_manifest),
                "dev_manifest": dev_manifest.name,
                "dev_manifest_sha256": file_sha256(dev_manifest),
                "train_samples": len(train_rows),
                "dev_samples": len(dev_rows),
                "sample_coverage": reference_coverage,
                "updates": reference_batches,
                "grad_scaler_state_dict": scaler.state_dict(),
            },
        )
        write_json(training_log_path, logs)
        reference_dev_predictions = collect_predictions(
            detector, dev_loader, device=device, amp_enabled=amp_enabled
        )
        reference_dev_metrics = summarize_predictions(
            reference_dev_predictions, threshold=float(config["threshold"])
        )
        write_json(evaluation_dir / "reference_dev_metrics.json", reference_dev_metrics)

        train_eval_loader = build_eval_loader(
            train_rows,
            image_size=int(config["image_size"]),
            batch_size=int(config["eval_batch_size"]),
            num_workers=int(config["num_workers"]),
            pin_memory=device.type == "cuda",
        )
        profile = _build_risk_profile(
            detector=detector,
            train_rows=train_rows,
            train_eval_loader=train_eval_loader,
            config=config,
            device=device,
            amp_enabled=amp_enabled,
            train_manifest=train_manifest,
            reference_checkpoint=reference_checkpoint,
            risk_profile_path=risk_profile_path,
        )

    if profile is None:
        train_eval_loader = build_eval_loader(
            train_rows,
            image_size=int(config["image_size"]),
            batch_size=int(config["eval_batch_size"]),
            num_workers=int(config["num_workers"]),
            pin_memory=device.type == "cuda",
        )
        profile = _build_risk_profile(
            detector=detector,
            train_rows=train_rows,
            train_eval_loader=train_eval_loader,
            config=config,
            device=device,
            amp_enabled=amp_enabled,
            train_manifest=train_manifest,
            reference_checkpoint=reference_checkpoint,
            risk_profile_path=risk_profile_path,
        )
    elif not args.resume:
        profile = load_risk_profile(risk_profile_path)
    _validate_profile(
        profile=profile, config=config, train_manifest=train_manifest,
        reference_checkpoint=reference_checkpoint
    )
    weights = weight_tensor(profile, device=device)
    print(f"Fixed risk profile verified: {len(train_rows)} Train predictions")

    bce_loader = build_training_loader(
        train_rows,
        image_size=int(config["image_size"]),
        batch_size=6,
        num_batches=total_bce_continuation,
        seed=int(config["bce_sampler_seed"]),
        num_workers=int(config["num_workers"]),
        pin_memory=device.type == "cuda",
        start_batch=completed_cycle * bce_per_cycle,
    )
    repair_loader = build_training_loader(
        train_rows,
        image_size=int(config["image_size"]),
        batch_size=6,
        num_batches=total_repair,
        seed=int(config["repair_sampler_seed"]),
        num_workers=int(config["num_workers"]),
        pin_memory=device.type == "cuda",
        start_batch=completed_cycle * repair_per_cycle,
    )
    bce_iterator = iter(bce_loader)
    repair_iterator = iter(repair_loader)
    bce_exposures: Counter[str] = Counter()
    repair_exposures: Counter[str] = Counter()
    repair_update = completed_cycle * repair_per_cycle
    global_update = reference_batches + completed_cycle * (bce_per_cycle + repair_per_cycle)

    for cycle in range(completed_cycle + 1, cycles + 1):
        for within_cycle in range(1, bce_per_cycle + 1):
            batch = next(bce_iterator)
            _record_batch_exposures(batch, bce_exposures)
            stats = _bce_update(
                detector=detector, optimizer=optimizer, scaler=scaler, batch=batch,
                device=device, amp_enabled=amp_enabled, threshold=float(config["threshold"])
            )
            global_update += 1
            logs.append({
                "phase": "bce_continuation", "cycle": cycle,
                "within_cycle_update": within_cycle, "global_update": global_update, **stats
            })

        for within_cycle in range(1, repair_per_cycle + 1):
            batch = next(repair_iterator)
            _record_batch_exposures(batch, repair_exposures)
            repair_update += 1
            stats = _joint_update(
                detector=detector,
                optimizer=optimizer,
                trainable_parameters=parameter_info["trainable_parameters"],
                batch=batch,
                weights=weights,
                config=config,
                device=device,
                amp_enabled=amp_enabled,
                repair_update=repair_update,
            )
            global_update += 1
            logs.append({
                "phase": "joint_repair", "cycle": cycle,
                "within_cycle_update": within_cycle, "repair_update": repair_update,
                "global_update": global_update, **stats
            })

        write_json(training_log_path, logs)
        save_checkpoint(
            final_checkpoint,
            detector=detector,
            optimizer=optimizer,
            config={
                **config, "completed_cycle": cycle,
                "global_updates": global_update, "device": str(device)
            },
            feature_dim=int(parameter_info["feature_dim"]),
            metadata={
                "stage": "alternating_visual_expert",
                "completed_cycle": cycle,
                "total_cycles": cycles,
                "global_updates": global_update,
                "reference_checkpoint_sha256": file_sha256(reference_checkpoint),
                "risk_profile_sha256": file_sha256(risk_profile_path),
                "train_manifest": train_manifest.name,
                "train_manifest_sha256": file_sha256(train_manifest),
                "dev_manifest": dev_manifest.name,
                "dev_manifest_sha256": file_sha256(dev_manifest),
                "optimizer_state_continuous": True,
                "bce_sampler_seed": int(config["bce_sampler_seed"]),
                "repair_sampler_seed": int(config["repair_sampler_seed"]),
                "grad_scaler_state_dict": scaler.state_dict(),
            },
        )
        print(
            f"[cycle {cycle}/{cycles}] completed; global_updates={global_update}; "
            f"checkpoint={final_checkpoint}"
        )
        # On a fresh run, validate actual consumed samples. On resume, the full
        # deterministic sampler plans above validate the complete run coverage.
        if not args.resume and cycle == cycles:
            validate_exact_exposures(
                train_rows, bce_exposures,
                expected_repeats=total_bce_continuation // expected_per_cell,
                phase_name="BCE continuation actual exposure",
            )
            validate_exact_exposures(
                train_rows, repair_exposures,
                expected_repeats=1,
                phase_name="joint repair actual exposure",
            )

    final_predictions = collect_predictions(
        detector, dev_loader, device=device, amp_enabled=amp_enabled
    )
    final_metrics = summarize_predictions(
        final_predictions, threshold=float(config["threshold"])
    )
    write_json(evaluation_dir / "final_dev_metrics.json", final_metrics)
    save_checkpoint(
        final_checkpoint,
        detector=detector,
        optimizer=optimizer,
        config={
            **config, "completed_cycle": cycles,
            "global_updates": reference_batches + total_bce_continuation + total_repair,
            "device": str(device)
        },
        feature_dim=int(parameter_info["feature_dim"]),
        metadata={
            "stage": "alternating_visual_expert_final",
            "completed_cycle": cycles,
            "total_cycles": cycles,
            "global_updates": reference_batches + total_bce_continuation + total_repair,
            "reference_checkpoint_sha256": file_sha256(reference_checkpoint),
            "risk_profile_sha256": file_sha256(risk_profile_path),
            "train_manifest": train_manifest.name,
            "train_manifest_sha256": file_sha256(train_manifest),
            "dev_manifest": dev_manifest.name,
            "dev_manifest_sha256": file_sha256(dev_manifest),
            "optimizer_state_continuous": True,
            "bce_sampler_seed": int(config["bce_sampler_seed"]),
            "repair_sampler_seed": int(config["repair_sampler_seed"]),
            "grad_scaler_state_dict": scaler.state_dict(),
            "final_dev_metrics": final_metrics,
        },
    )
    write_json(
        summary_path,
        {
            "protocol": "reference_then_14x_b100_b50",
            "reference_updates": reference_batches,
            "bce_continuation_updates": total_bce_continuation,
            "joint_repair_updates": total_repair,
            "total_parameter_updates": reference_batches + total_bce_continuation + total_repair,
            "train_samples": len(train_rows),
            "dev_samples": len(dev_rows),
            "sampling_coverage_plan": coverage_plan,
            "reference_dev_metrics": reference_dev_metrics,
            "final_dev_metrics": final_metrics,
            "reference_checkpoint_sha256": file_sha256(reference_checkpoint),
            "risk_profile_sha256": file_sha256(risk_profile_path),
            "final_checkpoint_sha256": file_sha256(final_checkpoint),
        },
    )
    write_json(training_log_path, logs)
    print("Training complete.")
    print(f"Reference updates: {reference_batches}")
    print(f"BCE continuation updates: {total_bce_continuation}")
    print(f"Joint repair updates: {total_repair}")
    print(f"Total updates: {reference_batches + total_bce_continuation + total_repair}")
    print(f"Final Dev metrics: {final_metrics}")


if __name__ == "__main__":
    main()
