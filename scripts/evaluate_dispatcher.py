#!/usr/bin/env python3
"""Evaluate the frozen DECD dispatcher on a scalar signals CSV."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from decd.dispatcher.abstention import UNCERTAIN  # noqa: E402
from decd.dispatcher.checkpoint import load_dispatcher_checkpoint  # noqa: E402
from decd.dispatcher.data import load_signals_csv  # noqa: E402
from decd.dispatcher.evaluation import evaluate_dispatcher  # noqa: E402

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "dispatcher_reported.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate fused DECD predictions with calibrated abstention."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--signals", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threshold", type=float)
    return parser.parse_args()


def _path(argument: Path | None, config_path: Path, configured: str) -> Path:
    if argument is not None:
        return argument.expanduser().resolve()
    return (config_path.parent.parent / configured).resolve()


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    signals_path = args.signals.expanduser().resolve()
    checkpoint_path = _path(
        args.checkpoint,
        args.config,
        config["paths"]["dispatcher_checkpoint"],
    )
    output_dir = _path(
        args.output_dir,
        args.config,
        config["paths"]["evaluation_dir"],
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    threshold = float(
        config["decision_threshold"] if args.threshold is None else args.threshold
    )

    dispatcher, payload = load_dispatcher_checkpoint(checkpoint_path, device=device)
    signals = load_signals_csv(signals_path)
    if signals["num_prompts"] != int(payload["num_prompts"]):
        raise ValueError("signals prompt count does not match the checkpoint")

    metrics = evaluate_dispatcher(
        dispatcher,
        signals,
        temperature_cv=float(payload["temperature_cv"]),
        temperature_mllm=float(payload["temperature_mllm"]),
        abstention_threshold_value=float(payload["abstention_threshold"]),
        device=device,
        decision_threshold=threshold,
    )
    rows_payload = metrics.pop("rows")
    predictions: list[dict] = []
    for index in range(metrics["n"]):
        committed = int(rows_payload["probability"][index] >= threshold)
        deferred = int(rows_payload["confidence"][index] < payload["abstention_threshold"])
        predictions.append(
            {
                "sample_id": signals["sample_ids"][index],
                "domain": signals["domains"][index],
                "label": int(signals["labels"][index]),
                "p_cv": float(rows_payload["p_cv"][index]),
                "p_mllm": float(rows_payload["p_mllm"][index]),
                "gate_weight_cv": float(rows_payload["weights"][index]),
                "fused_probability": float(rows_payload["probability"][index]),
                "routed_confidence": float(rows_payload["confidence"][index]),
                "prediction": UNCERTAIN if deferred else committed,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "predictions.csv", predictions)
    summary_rows = [{"scope": "overall", "domain": "all", **{
        key: value
        for key, value in metrics.items()
        if key not in {"per_domain", "selective"}
    }, **metrics["selective"]}]
    summary_rows.extend(
        {"scope": "domain", "domain": domain, **domain_metrics}
        for domain, domain_metrics in metrics["per_domain"].items()
    )
    _write_csv(output_dir / "metrics.csv", summary_rows)
    (output_dir / "metrics.json").write_text(
        json.dumps(
            {
                "signals": signals_path.name,
                "checkpoint": checkpoint_path.name,
                "temperature_cv": float(payload["temperature_cv"]),
                "temperature_mllm": float(payload["temperature_mllm"]),
                "abstention_threshold": float(payload["abstention_threshold"]),
                **metrics,
            },
            indent=2,
            default=float,
        ),
        encoding="utf-8",
    )
    print(
        f"evaluated {metrics['n']} samples: acc={metrics['accuracy']:.4f} "
        f"coverage={metrics['selective']['coverage']:.4f} "
        f"selective_acc={metrics['selective']['selective_accuracy']:.4f}"
    )
    print(f"results: {output_dir}")


if __name__ == "__main__":
    main()
