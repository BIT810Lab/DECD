"""Temperature calibration of frozen expert logits by validation NLL."""

from __future__ import annotations

from typing import Any

import numpy as np

GRID_MIN = 0.05
GRID_MAX = 5.0
GRID_STEP = 0.01
CLIP_EPSILON = 1.0e-7


def temperature_grid(
    minimum: float = GRID_MIN,
    maximum: float = GRID_MAX,
    step: float = GRID_STEP,
) -> np.ndarray:
    if minimum <= 0 or maximum <= minimum:
        raise ValueError("invalid temperature grid bounds")
    candidates = np.arange(minimum, maximum + step * 0.5, step)
    return np.round(candidates, decimals=4)


def temperature_nll(
    logits: np.ndarray,
    labels: np.ndarray,
    temperature: float,
) -> float:
    """Mean binary NLL of ``sigmoid(logit / T)``."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    probabilities = apply_temperature(logits, temperature)
    probabilities = np.clip(probabilities, CLIP_EPSILON, 1.0 - CLIP_EPSILON)
    labels = labels.astype(np.float64)
    return float(
        -np.mean(
            labels * np.log(probabilities)
            + (1.0 - labels) * np.log(1.0 - probabilities)
        )
    )


def fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    grid: np.ndarray | None = None,
) -> dict[str, Any]:
    """Grid-search ``T* = argmin_T NLL(T)``; report NLL at T=1 too."""

    logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    if logits.shape != labels.shape:
        raise ValueError("logits and labels must have the same length")
    candidates = temperature_grid() if grid is None else np.asarray(grid)
    nlls = np.array(
        [temperature_nll(logits, labels, float(t)) for t in candidates],
        dtype=np.float64,
    )
    best_index = int(np.argmin(nlls))
    return {
        "temperature": float(candidates[best_index]),
        "nll": float(nlls[best_index]),
        "uncalibrated_nll": temperature_nll(logits, labels, 1.0),
        "grid_size": int(candidates.size),
    }


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float64) / temperature))


def fit_expert_temperatures(
    *,
    cv_logits: np.ndarray,
    mllm_logits: np.ndarray,
    labels: np.ndarray,
    cv_temperature: float | None = None,
    mllm_temperature: float | None = None,
) -> dict[str, Any]:
    """Fit both experts on validation logits; MLLM across all prompt logits.

    A single shared ``T_mllm`` is fitted on the flattened ``N_val * K_p``
    prompt logits so every prompt verdict receives the same calibration.
    """

    cv_logits = np.asarray(cv_logits, dtype=np.float64).reshape(-1)
    mllm_logits = np.asarray(mllm_logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    if mllm_logits.ndim != 2 or mllm_logits.shape[0] != labels.shape[0]:
        raise ValueError("mllm_logits must have shape (N, K) matching labels")
    if cv_logits.shape[0] != labels.shape[0]:
        raise ValueError("cv_logits and labels must have the same length")

    if cv_temperature is None:
        cv_result = fit_temperature(cv_logits, labels)
        t_cv = cv_result["temperature"]
        cv_nll = cv_result["nll"]
    else:
        t_cv = float(cv_temperature)
        cv_nll = temperature_nll(cv_logits, labels, t_cv)

    if mllm_temperature is None:
        flat_logits = mllm_logits.reshape(-1)
        flat_labels = np.repeat(labels, mllm_logits.shape[1])
        mllm_result = fit_temperature(flat_logits, flat_labels)
        t_mllm = mllm_result["temperature"]
        mllm_nll = mllm_result["nll"]
    else:
        t_mllm = float(mllm_temperature)
        mllm_nll = temperature_nll(
            mllm_logits.reshape(-1),
            np.repeat(labels, mllm_logits.shape[1]),
            t_mllm,
        )

    return {
        "temperature_cv": float(t_cv),
        "temperature_mllm": float(t_mllm),
        "cv_nll": float(cv_nll),
        "mllm_nll": float(mllm_nll),
    }
