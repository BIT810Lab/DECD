"""Validation-quantile abstention threshold and selective decisions."""

from __future__ import annotations

import math

import numpy as np

UNCERTAIN = -1


def abstention_threshold(
    confidence: np.ndarray,
    budget: float,
) -> float:
    """Smallest ``tau`` deferring at least fraction ``budget`` of validation.

    Implements the paper's infimum definition, so tied confidence values may
    defer slightly more than the budget. ``budget == 0`` disables abstention.
    """

    values = np.asarray(confidence, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(values)):
        raise ValueError("confidence values must be finite")
    if not 0.0 <= budget < 1.0:
        raise ValueError("rejection budget must lie in [0, 1)")
    if values.size == 0:
        raise ValueError("confidence array is empty")
    if budget == 0.0:
        return 0.0

    ordered = np.sort(values)
    required = math.ceil(budget * values.size)
    if required >= values.size:
        # Only a threshold above every confidence value rejects all samples.
        return float("inf")
    return float(ordered[required])


def selective_predictions(
    probabilities: np.ndarray,
    confidence: np.ndarray,
    threshold: float,
    *,
    decision_threshold: float = 0.5,
) -> np.ndarray:
    """Labels with ``-1`` (``UNCERTAIN``) for deferred samples."""

    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    confidence = np.asarray(confidence, dtype=np.float64).reshape(-1)
    predictions = np.where(
        probabilities >= decision_threshold,
        1,
        0,
    ).astype(np.int64)
    predictions[confidence < threshold] = UNCERTAIN
    return predictions
