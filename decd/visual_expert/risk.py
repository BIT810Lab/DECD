"""Decision-margin statistics and fixed domain-label risk profiles."""

# SHA256 provenance binds the fixed risk profile to its training manifest and
# reference checkpoint. It is training validation, not dataset construction.

from __future__ import annotations

from collections import defaultdict
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .checkpoint import write_json
from .data import DOMAIN_TO_ID
from .sampler import EXPECTED_CELLS


def margin_key(value: float) -> str:
    return f"m{value:g}".replace(".", "p")


def signed_margin(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return np.where(labels == 1, logits, -logits)


def summarize_margins(
    logits: np.ndarray,
    labels: np.ndarray,
    gamma: float,
) -> dict[str, float | int]:
    if logits.ndim != 1 or logits.shape != labels.shape or not logits.size:
        raise ValueError('invalid risk input shape')
    if not np.all(np.isfinite(logits)) or not np.all(np.isin(labels, [0, 1])):
        raise ValueError('invalid/non-finite risk inputs')
    if not math.isfinite(gamma) or gamma < 0:
        raise ValueError('invalid risk margin')
    margins = signed_margin(logits, labels)
    deficits = np.maximum(float(gamma) - margins, 0.0)
    key = margin_key(gamma)
    return {
        "n": int(len(logits)),
        "mean_logit": float(np.mean(logits)),
        "mean_signed_margin": float(np.mean(margins)),
        "median_signed_margin": float(np.median(margins)),
        "signed_margin_q10": float(np.quantile(margins, 0.10)),
        f"risk_mean_{key}": float(np.mean(deficits)),
        f"risk_sq_mean_{key}": float(np.mean(np.square(deficits))),
        f"violation_rate_{key}": float(np.mean(deficits > 0)),
    }


def _normalize_labelwise(
    raw: np.ndarray,
    *,
    mode: str,
    alpha: float,
) -> np.ndarray:
    if not np.all(np.isfinite(raw)) or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("risk normalization alpha must be positive")
    weights = np.zeros_like(raw)
    for label in (0, 1):
        values = raw[:, label]
        if mode == "minmax":
            low = float(values.min())
            high = float(values.max())
            normalized = (
                np.ones_like(values)
                if high - low <= 1.0e-12
                else (values - low) / (high - low)
            )
            normalized = np.power(np.clip(normalized, 0.0, 1.0), alpha)
        elif mode == "maxnorm":
            high = float(values.max())
            normalized = (
                values / high if high > 1.0e-12 else np.ones_like(values)
            )
            normalized = np.power(np.clip(normalized, 0.0, 1.0), alpha)
        elif mode == "raw":
            normalized = np.power(np.maximum(values, 0.0), alpha)
        else:
            raise ValueError(f"unsupported risk normalization: {mode}")
        weights[:, label] = normalized
    return weights


def build_risk_profile(
    predictions: list[dict[str, Any]],
    *,
    gamma: float,
    risk_column: str,
    normalization: str,
    alpha: float,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a fixed six-cell risk profile from reference predictions."""

    if not math.isfinite(gamma) or gamma < 0:
        raise ValueError("gamma must be non-negative")

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        cell = (str(row["domain"]), int(row["label"]))
        if cell not in EXPECTED_CELLS:
            raise ValueError(f"unexpected domain-label cell: {cell}")
        grouped[cell].append(row)

    missing = [cell for cell in EXPECTED_CELLS if cell not in grouped]
    if missing:
        raise ValueError(f"risk manifest is missing cells: {missing}")

    statistics: dict[tuple[str, int], dict[str, Any]] = {}
    raw = np.zeros((len(DOMAIN_TO_ID), 2), dtype=np.float64)
    for cell in EXPECTED_CELLS:
        rows = grouped[cell]
        logits = np.asarray([row["logit"] for row in rows], dtype=np.float64)
        labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
        summary = summarize_margins(logits, labels, gamma)
        if risk_column not in summary:
            available = sorted(key for key in summary if key.startswith("risk_"))
            raise ValueError(
                f"risk column {risk_column!r} is unavailable; choose from "
                f"{available}"
            )
        statistics[cell] = summary
        raw[DOMAIN_TO_ID[cell[0]], cell[1]] = float(summary[risk_column])

    weights = _normalize_labelwise(raw, mode=normalization, alpha=alpha)
    cells: list[dict[str, Any]] = []
    for domain, label in EXPECTED_CELLS:
        domain_id = DOMAIN_TO_ID[domain]
        cells.append(
            {
                "domain": domain,
                "domain_id": domain_id,
                "label": label,
                "label_name": "Real" if label == 0 else "Fake",
                **statistics[(domain, label)],
                "selected_risk": float(raw[domain_id, label]),
                "repair_weight": float(weights[domain_id, label]),
            }
        )

    return {
        "format_version": 1,
        "gamma": float(gamma),
        "risk_column": risk_column,
        "normalization": normalization,
        "normalization_alpha": float(alpha),
        "raw_risk_matrix": raw.tolist(),
        "weight_matrix": weights.tolist(),
        "cells": cells,
        "provenance": provenance or {},
    }


def save_risk_profile(path: str | Path, profile: dict[str, Any]) -> None:
    write_json(path, profile)


def load_risk_profile(path: str | Path) -> dict[str, Any]:
    import json

    profile_path = Path(path).expanduser().resolve()
    if not profile_path.is_file():
        raise FileNotFoundError(profile_path)
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("risk profile root must be a mapping")
    weights = np.asarray(payload.get("weight_matrix"), dtype=np.float64)
    if weights.shape != (len(DOMAIN_TO_ID), 2):
        raise ValueError(
            f"risk weight matrix must have shape ({len(DOMAIN_TO_ID)}, 2)"
        )
    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("risk weights must be finite and non-negative")
    return payload


def validate_risk_profile(
    profile: dict[str, Any],
    *,
    risk_column: str,
    gamma: float,
    normalization: str,
    normalization_alpha: float,
    train_manifest_sha256: str,
    reference_checkpoint_sha256: str,
) -> None:
    """Require a risk profile to match the complete active training chain."""

    if profile.get('format_version') != 1:
        raise ValueError('unsupported risk profile format')
    raw = np.asarray(profile.get('raw_risk_matrix'), dtype=np.float64)
    weights = np.asarray(profile.get('weight_matrix'), dtype=np.float64)
    if (raw.shape != (len(DOMAIN_TO_ID), 2) or weights.shape != raw.shape
            or not np.all(np.isfinite(raw)) or np.any(raw < 0)
            or not np.all(np.isfinite(weights)) or np.any(weights < 0)):
        raise ValueError('invalid risk matrices')
    expected_weights = _normalize_labelwise(raw, mode=normalization, alpha=normalization_alpha)
    if not np.allclose(weights, expected_weights, rtol=0, atol=1e-12):
        raise ValueError('risk weights are inconsistent with the recorded raw risks')
    cells = profile.get('cells', [])
    if len(cells) != len(EXPECTED_CELLS):
        raise ValueError('risk profile must record all six cells')
    seen = set()
    for cell in cells:
        key = (cell.get('domain'), cell.get('label'))
        if key not in EXPECTED_CELLS or key in seen:
            raise ValueError('invalid or duplicate risk cell')
        seen.add(key)
        d, y = DOMAIN_TO_ID[key[0]], key[1]
        if cell.get('domain_id') != d or not isinstance(cell.get('n'), int) or cell['n'] <= 0:
            raise ValueError('invalid risk cell count/domain')
        for field, expected in [('selected_risk', raw[d, y]), ('repair_weight', weights[d, y])]:
            value = float(cell.get(field, float('nan')))
            if not math.isfinite(value) or abs(value - expected) > 1e-12:
                raise ValueError(f'inconsistent risk cell {field}')

    if str(profile.get("risk_column")) != str(risk_column):
        raise ValueError("risk profile column does not match the configuration")
    try:
        profile_gamma = float(profile["gamma"])
        profile_alpha = float(profile["normalization_alpha"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "risk profile is missing valid gamma or normalization alpha"
        ) from exc
    if not math.isfinite(profile_gamma) or abs(profile_gamma - float(gamma)) > 1.0e-12:
        raise ValueError("risk profile gamma does not match the configuration")
    if str(profile.get("normalization")) != str(normalization):
        raise ValueError(
            "risk profile normalization does not match the configuration"
        )
    if not math.isfinite(profile_alpha) or abs(profile_alpha - float(normalization_alpha)) > 1.0e-12:
        raise ValueError(
            "risk profile normalization alpha does not match the configuration"
        )

    provenance = profile.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("risk profile has no provenance mapping")
    manifest_hash = str(provenance.get("manifest_sha256", "")).lower()
    checkpoint_hash = str(
        provenance.get("reference_checkpoint_sha256", "")
    ).lower()
    if not manifest_hash or not checkpoint_hash:
        raise ValueError(
            "risk profile provenance is missing manifest or checkpoint SHA256"
        )
    if manifest_hash != str(train_manifest_sha256).lower():
        raise ValueError(
            "risk profile was built from a different training manifest"
        )
    if checkpoint_hash != str(reference_checkpoint_sha256).lower():
        raise ValueError(
            "risk profile was built from a different reference checkpoint"
        )


def weight_tensor(
    profile: dict[str, Any],
    *,
    device: torch.device,
) -> torch.Tensor:
    return torch.tensor(
        profile["weight_matrix"],
        dtype=torch.float32,
        device=device,
    )
