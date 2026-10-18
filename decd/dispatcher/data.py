"""Scalar signal manifests shared between the frozen experts and dispatcher.

The dispatcher trains solely on frozen scalar expert outputs. A signals CSV
needs the columns ``label`` and ``cv_logit``, plus one or more
``mllm_logit_<k>`` columns (``mllm_logit`` alone is accepted for K_p == 1).
``sample_id`` and ``domain`` are optional.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

import numpy as np

MLLM_COLUMN = re.compile(r"^mllm_logit_?(\d+)$")


def _parse_label(raw: str) -> int:
    value = float(raw)
    if value not in (0.0, 1.0):
        raise ValueError(f"label must be 0 or 1, got {raw!r}")
    return int(value)


def _ordered_mllm_columns(columns: list[str]) -> list[str]:
    indexed: list[tuple[int, str]] = []
    plain: list[str] = []
    for column in columns:
        match = MLLM_COLUMN.match(column)
        if match:
            indexed.append((int(match.group(1)), column))
        elif column == "mllm_logit":
            plain.append(column)
    indexed.sort(key=lambda item: item[0])
    ordered = [column for _, column in indexed]
    if ordered and plain:
        raise ValueError("mix ambiguous mllm_logit and mllm_logit_<k> columns")
    return ordered or plain


def load_signals_csv(path: str | Path) -> dict[str, Any]:
    csv_path = Path(path).expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"missing CSV header: {csv_path}")
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"signals manifest is empty: {csv_path}")

    for column in ("label", "cv_logit"):
        if column not in reader.fieldnames:
            raise ValueError(f"signals manifest is missing {column!r}")
    mllm_columns = _ordered_mllm_columns(list(reader.fieldnames))
    if not mllm_columns:
        raise ValueError("signals manifest needs at least one mllm_logit column")

    cv_logits = np.empty(len(rows), dtype=np.float64)
    mllm_logits = np.empty((len(rows), len(mllm_columns)), dtype=np.float64)
    labels = np.empty(len(rows), dtype=np.int64)
    sample_ids: list[str] = []
    domains: list[str] = []
    for index, row in enumerate(rows):
        labels[index] = _parse_label(row["label"])
        cv_logits[index] = float(row["cv_logit"])
        mllm_logits[index] = [float(row[column]) for column in mllm_columns]
        sample_ids.append(str(row.get("sample_id") or f"sample_{index:07d}"))
        domains.append(str(row.get("domain") or "all"))
    if not np.all(np.isfinite(cv_logits)) or not np.all(np.isfinite(mllm_logits)):
        raise ValueError("signal logits must be finite")

    return {
        "cv_logits": cv_logits,
        "mllm_logits": mllm_logits,
        "labels": labels,
        "sample_ids": sample_ids,
        "domains": domains,
        "num_prompts": len(mllm_columns),
        "mllm_columns": mllm_columns,
    }


def write_signals_csv(
    path: str | Path,
    *,
    cv_logits: np.ndarray,
    mllm_logits: np.ndarray,
    labels: np.ndarray,
    sample_ids: list[str] | None = None,
    domains: list[str] | None = None,
) -> None:
    cv_logits = np.asarray(cv_logits, dtype=np.float64).reshape(-1)
    mllm_logits = np.asarray(mllm_logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if mllm_logits.ndim == 1:
        mllm_logits = mllm_logits[:, None]
    if not (cv_logits.shape[0] == mllm_logits.shape[0] == labels.shape[0]):
        raise ValueError("signal arrays must share the sample dimension")

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["sample_id", "domain", "label", "cv_logit"]
    fields.extend(f"mllm_logit_{k}" for k in range(mllm_logits.shape[1]))
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in range(labels.shape[0]):
            writer.writerow(
                {
                    "sample_id": (
                        sample_ids[index] if sample_ids else f"sample_{index:07d}"
                    ),
                    "domain": domains[index] if domains else "all",
                    "label": int(labels[index]),
                    "cv_logit": float(cv_logits[index]),
                    **{
                        f"mllm_logit_{k}": float(mllm_logits[index, k])
                        for k in range(mllm_logits.shape[1])
                    },
                }
            )
