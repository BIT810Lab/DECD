"""Manifest handling, image preprocessing, datasets, and data loaders."""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset

from .sampler import DomainLabelBalancedBatchSampler

CLIP_MEAN = (0.48145466, 0.45782750, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

DOMAIN_TO_ID = {
    "Cartoon": 0,
    "Chart": 1,
    "Photo_and_poster": 2,
}
ID_TO_DOMAIN = {value: key for key, value in DOMAIN_TO_ID.items()}
DOMAIN_ALIASES = {
    "cartoon": "Cartoon",
    "chart": "Chart",
    "photo": "Photo_and_poster",
    "photo_and_poster": "Photo_and_poster",
    "photo & poster": "Photo_and_poster",
    "photoandposter": "Photo_and_poster",
}
PATH_FIELDS = (
    "image_path",
    "standardized_path",
    "original_image_path",
    "path",
    "filepath",
    "file_path",
)
DOMAIN_FIELDS = ("domain", "category", "source_domain", "dataset", "source")


def canonical_domain(value: object) -> str:
    key = str(value).strip().lower()
    if key not in DOMAIN_ALIASES:
        raise ValueError(
            f"unsupported domain {value!r}; expected Cartoon, Chart, or "
            "Photo_and_poster"
        )
    return DOMAIN_ALIASES[key]


def parse_label(row: dict[str, Any]) -> int:
    raw = str(row.get("label", "")).strip().lower()
    name = str(row.get("label_name", "")).strip().lower()
    name_to_label = {
        "real": 0,
        "human": 0,
        "fake": 1,
        "ai": 1,
        "generated": 1,
        "synthetic": 1,
    }

    if raw:
        try:
            numeric = float(raw)
        except ValueError as exc:
            raise ValueError(
                f"label must be numeric 0 or 1, got {raw!r}"
            ) from exc
        if numeric not in (0.0, 1.0):
            raise ValueError(f"label must be 0 or 1, got {raw!r}")
        label = int(numeric)
        if name:
            if name not in name_to_label:
                raise ValueError(f"unsupported label_name: {name!r}")
            if name_to_label[name] != label:
                raise ValueError(
                    f"conflicting label fields: label={raw!r}, "
                    f"label_name={name!r}"
                )
        return label

    if not name:
        raise ValueError("manifest row has neither label nor label_name")
    if name not in name_to_label:
        raise ValueError(f"unsupported label_name: {name!r}")
    return name_to_label[name]


def _choose_field(
    columns: Sequence[str],
    candidates: Sequence[str],
    explicit: str | None = None,
) -> str:
    if explicit:
        if explicit not in columns:
            raise ValueError(f"field {explicit!r} is absent from manifest")
        return explicit
    return next((candidate for candidate in candidates if candidate in columns), "")


def load_manifest(
    path: str | Path,
    *,
    require_domain: bool = True,
    path_field: str | None = None,
    domain_field: str | None = None,
    check_files: bool = True,
) -> list[dict[str, Any]]:
    """Read and normalize a CSV manifest.

    Relative image paths are resolved against the manifest directory. Paths
    remain internal to the returned rows and are not intended for result files.
    """

    manifest = Path(path).expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(manifest)

    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"missing CSV header: {manifest}")
        columns = list(reader.fieldnames)
        raw_rows = [dict(row) for row in reader]

    if not raw_rows:
        raise ValueError(f"manifest is empty: {manifest}")

    selected_path_field = _choose_field(columns, PATH_FIELDS, path_field)
    if not selected_path_field:
        raise ValueError(f"no supported image path field in {columns}")
    selected_domain_field = _choose_field(columns, DOMAIN_FIELDS, domain_field)
    if require_domain and not selected_domain_field:
        raise ValueError(f"no supported domain field in {columns}")

    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for index, row in enumerate(raw_rows):
        image_path = Path(str(row.get(selected_path_field, "")).strip())
        if not image_path.is_absolute():
            image_path = (manifest.parent / image_path).resolve()
        else:
            image_path = image_path.resolve()
        if check_files and not image_path.is_file():
            missing.append(str(image_path))
            continue

        label = parse_label(row)
        if selected_domain_field:
            raw_domain = str(row.get(selected_domain_field, "")).strip()
            domain = (
                canonical_domain(raw_domain)
                if require_domain
                else (raw_domain or "all")
            )
        else:
            domain = "all"

        normalized = dict(row)
        normalized.update(
            {
                "image_path": image_path,
                "domain": domain,
                "domain_id": DOMAIN_TO_ID.get(domain, -1),
                "label": label,
                "label_name": "Real" if label == 0 else "Fake",
                "manifest_index": index,
                "sample_id": str(
                    row.get("sample_id", f"sample_{index:07d}")
                ),
                "sha256": str(row.get("sha256", "")).strip().lower(),
            }
        )
        rows.append(normalized)

    if missing:
        preview = "\n".join(missing[:10])
        raise FileNotFoundError(
            f"{len(missing)} image files referenced by {manifest.name} are "
            f"missing; first entries:\n{preview}"
        )
    if not rows:
        raise ValueError(f"manifest has no valid samples: {manifest}")
    return rows


class LetterboxCLIPTransform:
    """Aspect-ratio-preserving resize, gray padding, and CLIP normalization."""

    def __init__(self, size: int = 224) -> None:
        if size <= 0:
            raise ValueError("image size must be positive")
        self.size = int(size)
        self.mean = torch.tensor(CLIP_MEAN, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(CLIP_STD, dtype=torch.float32).view(3, 1, 1)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = ImageOps.exif_transpose(image).convert("RGB")
        width, height = image.size
        if width <= 0 or height <= 0:
            raise ValueError(f"invalid image size: {image.size}")

        scale = min(self.size / width, self.size / height)
        new_width = max(1, round(width * scale))
        new_height = max(1, round(height * scale))
        image = image.resize(
            (new_width, new_height),
            Image.Resampling.BICUBIC,
        )

        canvas = Image.new("RGB", (self.size, self.size), color=(127, 127, 127))
        canvas.paste(
            image,
            ((self.size - new_width) // 2, (self.size - new_height) // 2),
        )
        array = np.asarray(canvas, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        return (tensor - self.mean) / self.std


class ManifestDataset(Dataset[dict[str, Any]]):
    def __init__(self, rows: list[dict[str, Any]], image_size: int = 224) -> None:
        self.rows = rows
        self.transform = LetterboxCLIPTransform(image_size)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        image_path = Path(row["image_path"])
        try:
            with Image.open(image_path) as opened:
                image = self.transform(opened)
        except Exception as exc:
            raise RuntimeError(f"failed to read image at manifest row {index}") from exc

        return {
            "image": image,
            "label": torch.tensor(row["label"], dtype=torch.float32),
            "domain_id": torch.tensor(row["domain_id"], dtype=torch.long),
            "domain": row["domain"],
            "sample_id": row["sample_id"],
            "manifest_index": torch.tensor(row["manifest_index"], dtype=torch.long),
            "sha256": row["sha256"],
        }


def build_training_loader(
    rows: list[dict[str, Any]],
    *,
    image_size: int,
    batch_size: int,
    num_batches: int,
    seed: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    dataset = ManifestDataset(rows, image_size)
    sampler = DomainLabelBalancedBatchSampler(
        rows,
        batch_size=batch_size,
        num_batches=num_batches,
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )


def build_eval_loader(
    rows: list[dict[str, Any]],
    *,
    image_size: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        ManifestDataset(rows, image_size),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
