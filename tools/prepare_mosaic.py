#!/usr/bin/env python3
"""Prepare a public, relative-path MOSAIC training manifest.

The input layout is:

    ROOT/{Cartoon,Chart,Photo}/{0_real,1_fake}/...

Source images are never changed. Accepted images are standardized as JPEG under
the output directory, while the manifest stores only relative paths.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
}
LAYOUT = {
    ("Cartoon", 0): Path("Cartoon/0_real"),
    ("Cartoon", 1): Path("Cartoon/1_fake"),
    ("Chart", 0): Path("Chart/0_real"),
    ("Chart", 1): Path("Chart/1_fake"),
    ("Photo_and_poster", 0): Path("Photo/0_real"),
    ("Photo_and_poster", 1): Path("Photo/1_fake"),
}
PATH_FIELDS = (
    "image_path",
    "standardized_path",
    "original_image_path",
    "path",
    "filepath",
    "file_path",
)
SHA_FIELDS = ("sha256", "image_sha256", "file_sha256")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan, de-duplicate, filter, standardize, and manifest MOSAIC data."
        )
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--exclude-manifest",
        type=Path,
        action="append",
        default=[],
        help="Dev/test manifest excluded by SHA256; may be repeated.",
    )
    parser.add_argument("--max-megapixels", type=float, default=50.0)
    parser.add_argument("--min-aspect-ratio", type=float, default=0.25)
    parser.add_argument("--max-aspect-ratio", type=float, default=4.0)
    parser.add_argument("--max-long-side", type=int, default=4096)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _choose_field(columns: list[str], candidates: tuple[str, ...]) -> str:
    return next((field for field in candidates if field in columns), "")


def load_exclusion_hashes(manifests: list[Path]) -> set[str]:
    hashes: set[str] = set()
    for manifest_value in manifests:
        manifest = manifest_value.expanduser().resolve()
        if not manifest.is_file():
            raise FileNotFoundError(manifest)
        with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = list(reader.fieldnames or [])
            sha_field = _choose_field(columns, SHA_FIELDS)
            path_field = _choose_field(columns, PATH_FIELDS)
            for row in reader:
                raw_hash = (
                    str(row.get(sha_field, "")).strip().lower()
                    if sha_field
                    else ""
                )
                if len(raw_hash) == 64:
                    hashes.add(raw_hash)
                    continue
                path_text = (
                    str(row.get(path_field, "")).strip()
                    if path_field
                    else ""
                )
                if not path_text:
                    continue
                image_path = Path(path_text)
                if not image_path.is_absolute():
                    image_path = (manifest.parent / image_path).resolve()
                if image_path.is_file():
                    hashes.add(sha256_file(image_path))
    return hashes


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _resize(image: Image.Image, max_long_side: int) -> Image.Image:
    width, height = image.size
    long_side = max(width, height)
    if long_side <= max_long_side:
        return image
    scale = max_long_side / long_side
    return image.resize(
        (
            max(1, round(width * scale)),
            max(1, round(height * scale)),
        ),
        Image.Resampling.LANCZOS,
    )


def standardize(
    source: Path,
    target: Path,
    *,
    max_long_side: int,
    jpeg_quality: int,
    overwrite: bool,
) -> tuple[int, int, str]:
    if target.is_file() and not overwrite:
        with Image.open(target) as existing:
            width, height = existing.size
        return width, height, sha256_file(target)

    target.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        image = _resize(image, max_long_side)
        width, height = image.size
        image.save(
            target,
            format="JPEG",
            quality=jpeg_quality,
            subsampling=0,
            optimize=False,
        )
    return width, height, sha256_file(target)


def main() -> None:
    args = parse_args()
    if args.max_megapixels <= 0:
        raise ValueError("max-megapixels must be positive")
    if not 0 < args.min_aspect_ratio < args.max_aspect_ratio:
        raise ValueError("aspect-ratio limits are invalid")
    if args.max_long_side <= 0:
        raise ValueError("max-long-side must be positive")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("jpeg-quality must be in [1, 100]")

    root = args.root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    manifest_dir = output_dir / "manifests"
    image_dir = output_dir / "images"
    audit_dir = output_dir / "audits"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)
    excluded_hashes = load_exclusion_hashes(args.exclude_manifest)

    candidates: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    source_index = 0
    for (domain, label), relative_pool in LAYOUT.items():
        pool = root / relative_pool
        if not pool.is_dir():
            raise FileNotFoundError(f"missing required pool: {relative_pool}")
        files = sorted(
            path
            for path in pool.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        for source in files:
            source_index += 1
            base = {
                "source_index": source_index,
                "domain": domain,
                "label": label,
                "label_name": "Real" if label == 0 else "Fake",
            }
            try:
                digest = sha256_file(source)
            except OSError as exc:
                excluded.append(
                    {
                        **base,
                        "reason": "hash_error",
                        "detail": type(exc).__name__,
                    }
                )
                continue
            if digest in excluded_hashes:
                excluded.append(
                    {
                        **base,
                        "sha256": digest,
                        "reason": "held_out_overlap",
                        "detail": "",
                    }
                )
                continue
            try:
                with Image.open(source) as opened:
                    image = ImageOps.exif_transpose(opened)
                    width, height = image.size
                    image.load()
            except Exception as exc:
                excluded.append(
                    {
                        **base,
                        "sha256": digest,
                        "reason": "unreadable_image",
                        "detail": type(exc).__name__,
                    }
                )
                continue

            megapixels = width * height / 1_000_000
            aspect_ratio = width / max(height, 1)
            if megapixels >= args.max_megapixels:
                excluded.append(
                    {
                        **base,
                        "sha256": digest,
                        "reason": "image_too_large",
                        "detail": f"megapixels={megapixels:.6f}",
                    }
                )
                continue
            if not args.min_aspect_ratio <= aspect_ratio <= args.max_aspect_ratio:
                excluded.append(
                    {
                        **base,
                        "sha256": digest,
                        "reason": "extreme_aspect_ratio",
                        "detail": f"aspect_ratio={aspect_ratio:.6f}",
                    }
                )
                continue
            candidates.append(
                {
                    **base,
                    "_source": source,
                    "sha256": digest,
                    "original_width": width,
                    "original_height": height,
                }
            )

    unique: list[dict[str, Any]] = []
    first_by_hash: dict[str, int] = {}
    for row in candidates:
        digest = row["sha256"]
        if digest in first_by_hash:
            excluded.append(
                {
                    key: value
                    for key, value in row.items()
                    if key != "_source"
                }
                | {
                    "reason": "exact_duplicate",
                    "detail": f"first_source_index={first_by_hash[digest]}",
                }
            )
        else:
            first_by_hash[digest] = int(row["source_index"])
            unique.append(row)

    accepted: list[dict[str, Any]] = []
    for row in unique:
        digest = str(row["sha256"])
        domain = str(row["domain"])
        label_name = str(row["label_name"])
        target = image_dir / domain / label_name / f"{digest}.jpg"
        try:
            width, height, standardized_hash = standardize(
                Path(row["_source"]),
                target,
                max_long_side=args.max_long_side,
                jpeg_quality=args.jpeg_quality,
                overwrite=args.overwrite,
            )
        except Exception as exc:
            excluded.append(
                {
                    key: value
                    for key, value in row.items()
                    if key != "_source"
                }
                | {
                    "reason": "standardization_error",
                    "detail": type(exc).__name__,
                }
            )
            continue

        relative_image = Path("..") / "images" / domain / label_name / target.name
        accepted.append(
            {
                "sample_id": f"{domain}_{label_name}_{digest[:16]}",
                "image_path": relative_image.as_posix(),
                "domain": domain,
                "label": int(row["label"]),
                "label_name": label_name,
                "sha256": digest,
                "standardized_sha256": standardized_hash,
                "width": width,
                "height": height,
                "split": "train",
            }
        )

    manifest_path = manifest_dir / "mosaic_train.csv"
    _write_csv(manifest_path, accepted)
    _write_csv(audit_dir / "excluded.csv", excluded)

    cell_counts = Counter(
        (row["domain"], row["label_name"]) for row in accepted
    )
    summary = {
        "seed": args.seed,
        "accepted_count": len(accepted),
        "excluded_count": len(excluded),
        "held_out_hash_count": len(excluded_hashes),
        "cell_counts": [
            {
                "domain": domain,
                "label_name": label_name,
                "count": cell_counts[(domain, label_name)],
            }
            for domain, label in LAYOUT
            for label_name in ["Real" if label == 0 else "Fake"]
        ],
        "filters": {
            "max_megapixels_exclusive": args.max_megapixels,
            "min_aspect_ratio_inclusive": args.min_aspect_ratio,
            "max_aspect_ratio_inclusive": args.max_aspect_ratio,
            "max_long_side": args.max_long_side,
            "jpeg_quality": args.jpeg_quality,
            "jpeg_subsampling": 0,
        },
        "manifest": "manifests/mosaic_train.csv",
        "note": "All accepted samples are retained; training performs balancing.",
    }
    (audit_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"accepted={len(accepted)}, excluded={len(excluded)}, "
        f"manifest={manifest_path}"
    )


if __name__ == "__main__":
    main()
