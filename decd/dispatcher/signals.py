"""Calibrated-scalar reliability signals and the six-dimensional vector."""

from __future__ import annotations

import numpy as np

RELIABILITY_DIM = 6
RELIABILITY_COLUMNS = (
    "p_cv",
    "p_mllm_mean",
    "margin_cv",
    "margin_mllm",
    "disagreement",
    "v_mllm",
)


def prompt_summary(prompt_probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Empirical mean and variance of the K_p calibrated prompt probs."""

    probs = np.asarray(prompt_probabilities, dtype=np.float64)
    if probs.ndim != 2:
        raise ValueError("prompt_probabilities must have shape (N, K)")
    mean = probs.mean(axis=1)
    variance = ((probs - mean[:, None]) ** 2).mean(axis=1)
    return mean, variance


def reliability_vector(
    cv_probabilities: np.ndarray,
    mllm_mean_probabilities: np.ndarray,
    mllm_variance: np.ndarray,
) -> np.ndarray:
    """Assemble Eq. (reliability): ``(N, 6)`` with every entry in ``[0, 1]``."""

    p_cv = np.asarray(cv_probabilities, dtype=np.float64).reshape(-1)
    p_mllm = np.asarray(mllm_mean_probabilities, dtype=np.float64).reshape(-1)
    variance = np.asarray(mllm_variance, dtype=np.float64).reshape(-1)
    if not (p_cv.shape == p_mllm.shape == variance.shape):
        raise ValueError("cv, mean-mllm, and variance arrays must align")

    vector = np.stack(
        [
            p_cv,
            p_mllm,
            2.0 * np.abs(p_cv - 0.5),
            2.0 * np.abs(p_mllm - 0.5),
            np.abs(p_cv - p_mllm),
            variance,
        ],
        axis=1,
    )
    if vector.shape[1] != RELIABILITY_DIM:
        raise AssertionError("reliability vector must be six-dimensional")
    return vector
