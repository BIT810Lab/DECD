#!/usr/bin/env python3
"""Stage II: calibrate frozen experts and train the dispatcher MLP."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from decd.dispatcher.checkpoint import save_dispatcher_checkpoint  # noqa: E402
from decd.dispatcher.data import load_signals_csv  # noqa: E402
from decd.dispatcher.evaluation import evaluate_dispatcher  # noqa: E402
from decd.dispatcher.training import train_dispatcher  # noqa: E402

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "dispatcher_reported.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit temperatures and train the confidence-aware dispatcher."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--train-signals", type=Path)
    parser.add_argument("--dev-signals", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--rejection-budget", type=float)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _path(argument: Path | None, config_path: Path, configured: str) -> Path:
    if argument is not None:
        return argument.expanduser().resolve()
    return (config_path.parent.parent / configured).resolve()


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    train_signals_path = _path(
        args.train_signals,
        args.config,
        config["signals"]["train"],
    )
    dev_signals_path = _path(
        args.dev_signals,
        args.config,
        config["signals"]["dev"],
    )
    output = _path(
        args.output,
        args.config,
        config["paths"]["dispatcher_checkpoint"],
    )
    device = torch.device(args.device or config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"output exists; pass --overwrite: {output}")

    train_signals = load_signals_csv(train_signals_path)
    dev_signals = load_signals_csv(dev_signals_path)
    result = train_dispatcher(
        train_signals,
        dev_signals,
        hidden_dim=int(config["hidden_dim"]),
        learning_rate=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
        batch_size=int(config["batch_size"]),
        epochs=int(args.epochs or config["epochs"]),
        early_stopping_patience=int(config["early_stopping_patience"]),
        rejection_budget=float(
            args.rejection_budget
            if args.rejection_budget is not None
            else config["rejection_budget"]
        ),
        device=device,
        seed=int(config["seed"]),
        fixed_temperature_cv=config.get("temperature_cv"),
        fixed_temperature_mllm=config.get("temperature_mllm"),
    )

    temperatures = result["temperatures"]
    dev_metrics = evaluate_dispatcher(
        result["dispatcher"],
        dev_signals,
        temperature_cv=temperatures["temperature_cv"],
        temperature_mllm=temperatures["temperature_mllm"],
        abstention_threshold_value=result["abstention_threshold"],
        device=device,
        decision_threshold=float(config["decision_threshold"]),
    )
    save_dispatcher_checkpoint(
        output,
        dispatcher=result["dispatcher"],
        config=config,
        temperature_cv=temperatures["temperature_cv"],
        temperature_mllm=temperatures["temperature_mllm"],
        abstention_threshold_value=result["abstention_threshold"],
        num_prompts=train_signals["num_prompts"],
        metrics={
            "dev_accuracy": dev_metrics["accuracy"],
            "dev_selective": dev_metrics["selective"],
        },
    )
    report = {
        "temperature_cv": temperatures["temperature_cv"],
        "temperature_mllm": temperatures["temperature_mllm"],
        "abstention_threshold": result["abstention_threshold"],
        "history": result["history"],
        "dev_accuracy": dev_metrics["accuracy"],
        "dev_macro_f1": dev_metrics["macro_f1"],
        "dev_selective": dev_metrics["selective"],
    }
    output.with_name("dispatcher_train_report.json").write_text(
        json.dumps(report, indent=2, default=float),
        encoding="utf-8",
    )
    print(
        "dispatcher trained: "
        f"T_cv={temperatures['temperature_cv']:.2f} "
        f"T_mllm={temperatures['temperature_mllm']:.2f} "
        f"tau_r={result['abstention_threshold']:.4f} "
        f"dev_acc={dev_metrics['accuracy']:.4f}"
    )
    print(f"saved dispatcher checkpoint: {output}")


if __name__ == "__main__":
    main()
