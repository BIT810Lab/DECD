#!/usr/bin/env python3
"""Build the fixed domain-label risk profile from a reference checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from decd.visual_expert.checkpoint import (  # noqa: E402
    load_yaml_config,
    resolve_project_path,
    restore_detector,
    validate_training_config,
)
from decd.visual_expert.data import build_eval_loader, load_manifest  # noqa: E402
from decd.visual_expert.model import collect_predictions, set_determinism  # noqa: E402
from decd.visual_expert.risk import (  # noqa: E402
    build_risk_profile,
    save_risk_profile,
)

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "visual_expert_reported.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a fixed risk profile from reference predictions."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device")
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
    manifest = _path(
        args.manifest,
        args.config,
        config["manifests"]["train"],
    )
    checkpoint = _path(
        args.checkpoint,
        args.config,
        config["paths"]["reference_checkpoint"],
    )
    output = _path(
        args.output,
        args.config,
        config["paths"]["risk_profile"],
    )
    device = torch.device(args.device or config["device"])
    if output.exists():
        raise FileExistsError('risk profile already exists; choose a new output')
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    set_determinism(int(config["seed"]))
    rows = load_manifest(manifest, require_domain=True, role='train')
    detector, _, _, checkpoint_metadata = restore_detector(
        checkpoint,
        device=device,
        restore_optimizer=False,
        expected_config=config,
        require_reference=True,
        train_manifest=manifest,
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
    profile = build_risk_profile(
        predictions,
        gamma=float(config["gamma"]),
        risk_column=str(config["risk_column"]),
        normalization=str(config["risk_normalization"]),
        alpha=float(config["risk_normalization_alpha"]),
        provenance={
            "manifest": manifest.name,
            "reference_checkpoint": checkpoint.name,
        },
    )
    save_risk_profile(output, profile)
    print(
        f"saved {profile['risk_column']} profile for "
        f"{len(profile['cells'])} cells: {output}"
    )


if __name__ == "__main__":
    main()
