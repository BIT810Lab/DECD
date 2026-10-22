#!/usr/bin/env python3
"""Evaluate a frozen Visual Expert checkpoint."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from decd.visual_expert.checkpoint import (  # noqa: E402
    load_yaml_config,
    resolve_project_path,
    restore_detector,
    write_json,
)
from decd.visual_expert.data import build_eval_loader, load_manifest  # noqa: E402
from decd.visual_expert.model import (  # noqa: E402
    collect_predictions,
    set_determinism,
    summarize_predictions,
)

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "visual_expert_reported.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the shared-head Visual Expert; domain labels are optional."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--path-field")
    parser.add_argument("--domain-field")
    parser.add_argument('--overwrite', action='store_true',
                        help='Explicitly allow overwriting existing evaluation files.')
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


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    manifest = _path(
        args.manifest,
        args.config,
        # Alternating-training configs only require Train/Dev entries.
        # An explicit --manifest still takes precedence over this public default.
        config.get("manifests", {}).get("eval", "data/manifests/mosaic_test.csv"),
    )
    checkpoint = _path(
        args.checkpoint,
        args.config,
        config["paths"]["visual_expert_checkpoint"],
    )
    output_dir = _path(
        args.output_dir,
        args.config,
        config["paths"]["evaluation_dir"],
    )
    if output_dir.exists() and not args.overwrite and any(
        (output_dir / name).exists() for name in ('predictions.csv', 'metrics.csv', 'metrics.json')
    ):
        raise FileExistsError('evaluation files exist; use a new directory or --overwrite')
    device = torch.device(args.device or config["device"])
    threshold = float(
        config["threshold"] if args.threshold is None else args.threshold
    )
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    set_determinism(int(config["seed"]))
    rows = load_manifest(
        manifest,
        require_domain=False,
        path_field=args.path_field,
        domain_field=args.domain_field,
    )
    detector, _, _, checkpoint_metadata = restore_detector(
        checkpoint,
        device=device,
        restore_optimizer=False,
    )
    for parameter in detector.parameters():
        parameter.requires_grad = False
    loader = build_eval_loader(
        rows,
        image_size=int(checkpoint_metadata["image_size"]),
        batch_size=int(config["eval_batch_size"]),
        num_workers=int(config["num_workers"]),
        pin_memory=device.type == "cuda",
    )
    predictions = collect_predictions(
        detector,
        loader,
        device=device,
        amp_enabled=bool(config["amp"]) and device.type == "cuda",
    )
    metrics = summarize_predictions(predictions, threshold=threshold)

    public_predictions: list[dict[str, Any]] = []
    for row in predictions:
        predicted = int(row["probability"] >= threshold)
        public_predictions.append(
            {
                **row,
                "label_name": "Real" if row["label"] == 0 else "Fake",
                "prediction": predicted,
                "prediction_name": "Real" if predicted == 0 else "Fake",
                "correct": int(predicted == row["label"]),
                "threshold": threshold,
            }
        )
    summary_rows = [
        {"scope": "overall", "domain": "all", **metrics["overall"]}
    ]
    summary_rows.extend(
        {
            "scope": "domain",
            "domain": domain,
            **domain_metrics,
        }
        for domain, domain_metrics in metrics["per_domain"].items()
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "predictions.csv", public_predictions)
    _write_csv(output_dir / "metrics.csv", summary_rows)
    write_json(
        output_dir / "metrics.json",
        {
            "manifest": manifest.name,
            "checkpoint": checkpoint_metadata,
            "threshold": threshold,
            **metrics,
        },
    )
    print(f"evaluated {len(rows)} samples; results: {output_dir}")


if __name__ == "__main__":
    main()
