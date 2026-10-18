"""Fusion evaluation: full-coverage metrics and calibrated abstention."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)

from .abstention import UNCERTAIN, selective_predictions
from .model import fused_probability, routed_confidence
from .training import prepare_inputs


@torch.no_grad()
def score_signals(
    dispatcher: torch.nn.Module,
    signals: dict[str, Any],
    *,
    temperature_cv: float,
    temperature_mllm: float,
    device: torch.device | str = "cpu",
) -> dict[str, np.ndarray]:
    """Return gate, fused probability, and routed confidence for a split."""

    inputs = prepare_inputs(
        signals,
        temperature_cv=temperature_cv,
        temperature_mllm=temperature_mllm,
    )
    dispatcher.eval()
    gate_logits = dispatcher(inputs["reliability"].to(torch.device(device)))
    fused, weights = fused_probability(
        gate_logits,
        inputs["p_cv"].to(device),
        inputs["p_mllm"].to(device),
    )
    confidence = routed_confidence(weights, fused)
    return {
        "weights": weights.float().cpu().numpy(),
        "probability": fused.float().cpu().numpy(),
        "confidence": confidence.float().cpu().numpy(),
        "labels": signals["labels"].astype(np.int64),
        "p_cv": inputs["p_cv"].numpy(),
        "p_mllm": inputs["p_mllm"].numpy(),
    }


def selective_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float]:
    covered = predictions != UNCERTAIN
    coverage = float(np.mean(covered))
    if not np.any(covered):
        return {"coverage": 0.0, "selective_accuracy": float("nan"), "selective_risk": float("nan")}
    correct = predictions[covered] == labels[covered]
    selective_accuracy = float(np.mean(correct))
    return {
        "coverage": coverage,
        "selective_accuracy": selective_accuracy,
        "selective_risk": float(1.0 - selective_accuracy),
        "abstained": int(np.sum(~covered)),
    }


def evaluate_dispatcher(
    dispatcher: torch.nn.Module,
    signals: dict[str, Any],
    *,
    temperature_cv: float,
    temperature_mllm: float,
    abstention_threshold_value: float,
    device: torch.device | str = "cpu",
    decision_threshold: float = 0.5,
) -> dict[str, Any]:
    scored = score_signals(
        dispatcher,
        signals,
        temperature_cv=temperature_cv,
        temperature_mllm=temperature_mllm,
        device=device,
    )
    labels = scored["labels"]
    probability = scored["probability"]
    committed_predictions = (probability >= decision_threshold).astype(np.int64)

    metrics: dict[str, Any] = {
        "n": int(labels.size),
        "accuracy": float(accuracy_score(labels, committed_predictions)),
        "macro_f1": float(
            f1_score(labels, committed_predictions, average="macro", zero_division=0)
        ),
        "ap": float(average_precision_score(labels, probability)),
        "auroc": float(roc_auc_score(labels, probability)),
        "mean_gate_weight_cv": float(np.mean(scored["weights"])),
        "mean_confidence": float(np.mean(scored["confidence"])),
    }
    predictions = selective_predictions(
        probability,
        scored["confidence"],
        abstention_threshold_value,
        decision_threshold=decision_threshold,
    )
    metrics["selective"] = selective_metrics(labels, predictions)

    per_domain: dict[str, dict[str, Any]] = {}
    domains = signals.get("domains")
    if domains:
        domains_array = np.asarray(domains)
        for domain in sorted(domains_array.tolist()):
            mask = domains_array == domain
            domain_predictions = selective_predictions(
                probability[mask],
                scored["confidence"][mask],
                abstention_threshold_value,
                decision_threshold=decision_threshold,
            )
            per_domain[str(domain)] = {
                "accuracy": float(
                    accuracy_score(labels[mask], (probability[mask] >= 0.5).astype(int))
                ),
                **selective_metrics(labels[mask], domain_predictions),
            }
    metrics["per_domain"] = per_domain
    metrics["rows"] = {
        key: scored[key]
        for key in ("weights", "probability", "confidence", "p_cv", "p_mllm")
    }
    return metrics
