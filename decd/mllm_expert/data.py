"""Manifest loading and image dataset for frozen-feature extraction.

The semantic expert consumes the same manifest CSV convention as the visual
expert (``image_path`` / ``label`` / optional ``domain``), but this module is
self-contained so the two evidence experts stay decoupled. The MLLM processor
does its own image preprocessing, so samples yield RGB PIL images.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset

PATH_FIELDS = ("image_path", "standardized_path", "path", "file_path", "filepath")
LABEL_NAMES = {
    "real": 0,
    "human": 0,
    "fake": 1,
    "ai": 1,
    "generated": 1,
    "synthetic": 1,
}


def parse_label(row: dict[str, Any]) -> int:
    raw = str(row.get("label", "")).strip().lower()
    name = str(row.get("label_name", "")).strip().lower()
    if raw:
        numeric = float(raw)
        if numeric not in (0.0, 1.0):
            raise ValueError(f"label must be 0 or 1, got {raw!r}")
        label = int(numeric)
        if name:
            if name not in LABEL_NAMES:
                raise ValueError(f"unsupported label_name: {name!r}")
            if LABEL_NAMES[name] != label:
                raise ValueError(
                    f"conflicting label fields: label={raw!r}, label_name={name!r}"
                )
        return label
    if name not in LABEL_NAMES:
        raise ValueError("manifest row has neither a numeric label nor a label_name")
    return LABEL_NAMES[name]


def load_mllm_manifest(
    path: str | Path,
    *,
    check_files: bool = True,
) -> list[dict[str, Any]]:
    manifest = Path(path).expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        raw_rows = [dict(row) for row in reader]
    if not raw_rows:
        raise ValueError(f"manifest is empty: {manifest}")
    path_field = next(
        (field for field in PATH_FIELDS if field in raw_rows[0]),
        "",
    )
    if not path_field:
        raise ValueError(f"no supported image path field in {manifest}")

    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for index, row in enumerate(raw_rows):
        image_path = Path(str(row.get(path_field, "")).strip())
        if not image_path.is_absolute():
            image_path = (manifest.parent / image_path).resolve()
        else:
            image_path = image_path.resolve()
        if check_files and not image_path.is_file():
            missing.append(str(image_path))
            continue
        label = parse_label(row)
        rows.append(
            {
                "image_path": image_path,
                "label": label,
                "label_name": "Real" if label == 0 else "Fake",
                "domain": str(row.get("domain", "all") or "all"),
                "sample_id": str(
                    row.get("sample_id", f"sample_{index:07d}")
                ),
                "manifest_index": index,
            }
        )
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} image files are missing; first entries:\n"
            + "\n".join(missing[:10])
        )
    if not rows:
        raise ValueError(f"manifest has no valid samples: {manifest}")
    return rows


class MLLMImageDataset(Dataset[dict[str, Any]]):
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        try:
            with Image.open(row["image_path"]) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"failed to read image at manifest row {index}") from exc
        return {
            "image": image,
            "label": torch.tensor(row["label"], dtype=torch.float32),
            "sample_id": row["sample_id"],
            "domain": row["domain"],
            "manifest_index": row["manifest_index"],
        }


def build_image_loader(
    rows: list[dict[str, Any]],
    *,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        MLLMImageDataset(rows),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=_pil_collate,
    )


def _pil_collate(
    items: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "image": [item["image"] for item in items],
        "label": torch.stack([item["label"] for item in items]),
        "sample_id": [item["sample_id"] for item in items],
        "domain": [item["domain"] for item in items],
        "manifest_index": [item["manifest_index"] for item in items],
    }
