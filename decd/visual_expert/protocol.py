"""Validation helpers for full-data Visual Expert training protocols."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import math
from pathlib import Path
from typing import Any

from .sampler import EXPECTED_CELLS


def validate_training_schedule(config: Mapping[str, Any]) -> None:
    expected = {"batch_size": 6, "expected_train_per_cell": 700,
                "expected_dev_per_cell": 100, "reference_batches": 700,
                "cycles": 14, "bce_batches_per_cycle": 100,
                "repair_batches_per_cycle": 50}
    for key, value in expected.items():
        if type(config.get(key)) is not int or config[key] != value:
            raise ValueError(f"Full-data schedule requires {key}={value}")
    if str(config.get("repair_label_side", "")).lower() != "real":
        raise ValueError("Full-data schedule requires real-side repair")


def validate_train_dev_integrity(train_rows, dev_rows) -> dict[str, int]:
    """Check existing paths, roles and supplied groups without reading image bytes."""
    path_roles, group_roles = {}, {}
    for role, rows in (("train", train_rows), ("dev", dev_rows)):
        for row in rows:
            if row.get("split") and row["split"] != role:
                raise ValueError(f"Manifest role mismatch: expected {role}")
            path = Path(row["image_path"]).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            if path in path_roles:
                raise ValueError("Duplicate/cross-split Train/Dev image path")
            path_roles[path] = role
            group = str(row.get("group_id", "")).strip()
            if group and group_roles.setdefault(group, role) != role:
                raise ValueError("Train/Dev source group overlap")
    return {"verified_image_paths": len(path_roles), "train_dev_path_overlap": 0,
            "available_group_overlap": 0}


def validate_resume_contract(payload, config) -> int:
    """Validate saved configuration, input filenames and complete AdamW state."""
    stored, meta = payload.get("config", {}), payload.get("metadata", {})
    keys = ("seed", "model_name", "pretrained", "trainable_blocks", "image_size",
            "batch_size", "backbone_lr", "head_lr", "weight_decay", "amp",
            "threshold", "reference_batches", "cycles", "bce_batches_per_cycle",
            "repair_batches_per_cycle", "bce_sampler_seed", "repair_sampler_seed",
            "gamma", "repair_lambda", "ramp_batches", "rho_max", "gradient_epsilon",
            "repair_label_side", "risk_column", "risk_normalization", "risk_normalization_alpha")
    # Checkpoint serialization stores absolute initialization paths as basenames.
    for key in keys:
        current = config.get(key)
        if key == "pretrained" and isinstance(current, str) and Path(current).is_absolute():
            current = Path(current).name
        if stored.get(key) != current:
            raise ValueError(f"Resume config mismatch: {key}")
    for role in ("train", "dev"):
        expected = Path(config["manifests"][role]).name
        if meta.get(f"{role}_manifest") != expected:
            raise ValueError(f"Resume manifest name mismatch: {role}")
    stage = meta.get("stage")
    if stage == "reference":
        completed, updates = 0, int(config["reference_batches"])
        if meta.get("updates") != updates:
            raise ValueError("Incomplete reference checkpoint")
    elif stage in ("alternating_visual_expert", "alternating_visual_expert_final"):
        completed = meta.get("completed_cycle")
        if type(completed) is not int or not 1 <= completed <= int(config["cycles"]):
            raise ValueError("Invalid completed cycle")
        updates = int(config["reference_batches"]) + completed * (int(config["bce_batches_per_cycle"]) + int(config["repair_batches_per_cycle"]))
        if stored.get("completed_cycle") != completed or meta.get("global_updates") != updates:
            raise ValueError("Resume cycle/update metadata mismatch")
        for key in ("reference_checkpoint", "risk_profile"):
            recorded = meta.get(key)
            if recorded and recorded != Path(config["paths"][key]).name:
                raise ValueError(f"Resume input filename mismatch: {key}")
    else:
        raise ValueError(f"Unsupported resume stage: {stage}")
    state = payload.get("optimizer_state_dict")
    if not isinstance(state, dict) or not state.get("param_groups"):
        raise ValueError("Resume checkpoint is missing AdamW state")
    ids = [index for group in state["param_groups"] for index in group["params"]]
    moments = state.get("state", {})
    if not ids or len(ids) != len(set(ids)) or set(ids) != set(moments):
        raise ValueError("Incomplete AdamW parameter state")
    for index in ids:
        item = moments[index]
        if not all(key in item for key in ("step", "exp_avg", "exp_avg_sq")):
            raise ValueError("Missing AdamW moments")
        step = float(item["step"])
        if not math.isfinite(step) or step != updates:
            raise ValueError("AdamW step does not match completed updates")
        if item["exp_avg"].shape != item["exp_avg_sq"].shape:
            raise ValueError("AdamW moment shapes differ")
        if not bool(item["exp_avg"].isfinite().all()) or not bool(item["exp_avg_sq"].isfinite().all()):
            raise ValueError("Non-finite AdamW moments")
    return completed


def validate_balanced_split(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_per_cell: int,
    split_name: str,
) -> dict[str, int]:
    """Require the six domain-label cells to have exact expected sizes."""
    if expected_per_cell <= 0:
        raise ValueError("expected_per_cell must be positive")
    if not rows:
        raise ValueError(f"{split_name} manifest is empty")

    sample_ids = [str(row["sample_id"]) for row in rows]
    duplicates = [sample_id for sample_id, count in Counter(sample_ids).items() if count > 1]
    if duplicates:
        raise ValueError(
            f"{split_name} manifest has duplicate sample_id values; "
            f"examples: {duplicates[:5]}"
        )

    counts = Counter((str(row["domain"]), int(row["label"])) for row in rows)
    expected = {cell: expected_per_cell for cell in EXPECTED_CELLS}
    if dict(counts) != expected:
        details = ", ".join(
            f"{domain}/{'Real' if label == 0 else 'Fake'}="
            f"{counts.get((domain, label), 0)} (expected {expected_per_cell})"
            for domain, label in EXPECTED_CELLS
        )
        raise ValueError(
            f"{split_name} manifest does not match the required balanced split: "
            f"{details}; total={len(rows)}, expected={expected_per_cell * len(EXPECTED_CELLS)}"
        )
    return {f"{domain}:{label}": counts[(domain, label)] for domain, label in EXPECTED_CELLS}


def validate_exact_exposures(
    rows: Sequence[Mapping[str, Any]],
    exposure_counts: Mapping[str, int],
    *,
    expected_repeats: int,
    phase_name: str,
) -> dict[str, int]:
    """Require each unique manifest sample to be seen an exact number of times."""
    expected_ids = {str(row["sample_id"]) for row in rows}
    actual_ids = set(exposure_counts)
    missing = sorted(expected_ids - actual_ids)
    unexpected = sorted(actual_ids - expected_ids)
    wrong = {
        sample_id: int(exposure_counts.get(sample_id, 0))
        for sample_id in expected_ids
        if int(exposure_counts.get(sample_id, 0)) != expected_repeats
    }
    if missing or unexpected or wrong:
        examples = list(wrong.items())[:5]
        raise RuntimeError(
            f"{phase_name} sample-coverage validation failed: "
            f"missing_ids={len(missing)}, unexpected_ids={len(unexpected)}, "
            f"wrong_exposure_counts={len(wrong)}, examples={examples}"
        )
    return {
        "unique_samples": len(expected_ids),
        "total_exposures": sum(int(v) for v in exposure_counts.values()),
        "expected_exposures_per_sample": expected_repeats,
    }
