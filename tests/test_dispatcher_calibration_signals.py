"""Tests for temperature calibration and the six-dimensional reliability vector."""

import numpy as np
import pytest

from decd.dispatcher.calibration import (
    apply_temperature,
    fit_expert_temperatures,
    fit_temperature,
    temperature_nll,
)
from decd.dispatcher.signals import RELIABILITY_DIM, prompt_summary, reliability_vector


def test_apply_temperature_matches_sigmoid():
    logits = np.linspace(-3, 3, 7)
    np.testing.assert_allclose(
        apply_temperature(logits, 1.0),
        1 / (1 + np.exp(-logits)),
        rtol=1e-12,
    )
    assert np.isclose(apply_temperature(np.array([0.0]), 2.0)[0], 0.5)
    with pytest.raises(ValueError):
        apply_temperature(logits, 0.0)


def test_nll_at_temperature_one_manual():
    logits = np.array([0.0, 2.0])
    labels = np.array([0, 1])
    probs = 1 / (1 + np.exp(-logits))
    expected = -np.mean(
        [np.log(1 - probs[0]), np.log(probs[1])]
    )
    assert temperature_nll(logits, labels, 1.0) == pytest.approx(expected)


def test_fit_recovers_rescaling_temperature():
    # Labels are drawn from Bernoulli(sigmoid(s)); doubling the stored logits
    # makes T* = 2 the only temperature that recovers the true probabilities.
    rng = np.random.default_rng(0)
    true_logits = rng.normal(scale=1.5, size=4000)
    labels = (rng.random(4000) < 1 / (1 + np.exp(-true_logits))).astype(int)
    stored_logits = 2.0 * true_logits
    grid = np.array([0.5, 1.0, 2.0, 3.0])
    result = fit_temperature(stored_logits, labels, grid=grid)
    assert result["temperature"] == pytest.approx(2.0)
    assert result["nll"] <= result["uncalibrated_nll"]


def test_fit_expert_temperatures_shapes_and_overrides():
    rng = np.random.default_rng(1)
    n = 50
    labels = rng.integers(0, 2, n)
    cv = rng.normal(size=n)
    mllm = rng.normal(size=(n, 3))
    result = fit_expert_temperatures(
        cv_logits=cv,
        mllm_logits=mllm,
        labels=labels,
        cv_temperature=0.85,
        mllm_temperature=0.70,
    )
    assert result["temperature_cv"] == pytest.approx(0.85)
    assert result["temperature_mllm"] == pytest.approx(0.70)

    fitted = fit_expert_temperatures(
        cv_logits=cv,
        mllm_logits=mllm,
        labels=labels,
    )
    assert 0.05 <= fitted["temperature_cv"] <= 5.0
    assert 0.05 <= fitted["temperature_mllm"] <= 5.0


def test_prompt_summary():
    probs = np.array([[0.2, 0.4, 0.6], [0.1, 0.1, 0.4]])
    mean, variance = prompt_summary(probs)
    np.testing.assert_allclose(mean, [0.4, 0.2])
    np.testing.assert_allclose(
        variance,
        [np.mean((np.array([0.2, 0.4, 0.6]) - 0.4) ** 2),
         np.mean((np.array([0.1, 0.1, 0.4]) - 0.2) ** 2)],
    )


def test_reliability_vector_hand_values():
    vector = reliability_vector(
        np.array([0.8]),
        np.array([0.3]),
        np.array([0.04]),
    )
    assert vector.shape == (1, RELIABILITY_DIM)
    np.testing.assert_allclose(
        vector[0],
        np.array([0.8, 0.3, 0.6, 0.4, 0.5, 0.04]),
        rtol=1e-12,
    )


def test_reliability_vector_is_unit_interval():
    rng = np.random.default_rng(2)
    p_cv = rng.random(100)
    p_mllm = rng.random(100)
    variance = rng.random(100) * 0.25
    vector = reliability_vector(p_cv, p_mllm, variance)
    assert np.all(vector >= 0.0)
    assert np.all(vector <= 1.0)
