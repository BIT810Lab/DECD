"""Signal-CSV I/O and the full Stage-II training/evaluation/checkpoint path."""

import csv

import numpy as np
import pytest
import torch

from decd.dispatcher.checkpoint import (
    load_dispatcher_checkpoint,
    save_dispatcher_checkpoint,
)
from decd.dispatcher.data import load_signals_csv, write_signals_csv
from decd.dispatcher.evaluation import evaluate_dispatcher
from decd.dispatcher.training import train_dispatcher


def _write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_load_indexed_mllm_columns(tmp_path):
    path = tmp_path / "signals.csv"
    rows = [
        {"sample_id": "a", "domain": "Cartoon", "label": 0,
         "cv_logit": -0.2, "mllm_logit_0": -0.1,
         "mllm_logit_1": -0.3, "mllm_logit_2": 0.1},
        {"sample_id": "b", "domain": "Chart", "label": 1,
         "cv_logit": 1.2, "mllm_logit_0": 0.8,
         "mllm_logit_1": 1.1, "mllm_logit_2": 0.4},
    ]
    _write_csv(
        path,
        rows,
        ["sample_id", "domain", "label", "cv_logit",
         "mllm_logit_0", "mllm_logit_1", "mllm_logit_2"],
    )
    signals = load_signals_csv(path)
    assert signals["mllm_logits"].shape == (2, 3)
    assert signals["labels"].tolist() == [0, 1]
    np.testing.assert_allclose(signals["cv_logits"], [-0.2, 1.2])
    np.testing.assert_allclose(signals["mllm_logits"][1], [0.8, 1.1, 0.4])
    assert signals["sample_ids"] == ["a", "b"]
    assert signals["domains"] == ["Cartoon", "Chart"]


def test_plain_single_mllm_column(tmp_path):
    path = tmp_path / "signals.csv"
    _write_csv(
        path,
        [{"label": 1, "cv_logit": 0.5, "mllm_logit": 0.2}],
        ["label", "cv_logit", "mllm_logit"],
    )
    signals = load_signals_csv(path)
    assert signals["num_prompts"] == 1
    assert signals["mllm_logits"].shape == (1, 1)


def test_missing_and_mixed_columns_raise(tmp_path):
    missing = tmp_path / "missing.csv"
    _write_csv(missing, [{"label": 0, "mllm_logit_0": 0.1}], ["label", "mllm_logit_0"])
    with pytest.raises(ValueError):
        load_signals_csv(missing)

    no_mllm = tmp_path / "no_mllm.csv"
    _write_csv(no_mllm, [{"label": 0, "cv_logit": 0.1}], ["label", "cv_logit"])
    with pytest.raises(ValueError):
        load_signals_csv(no_mllm)

    mixed = tmp_path / "mixed.csv"
    _write_csv(
        mixed,
        [{"label": 0, "cv_logit": 0.1, "mllm_logit": 0.2, "mllm_logit_0": 0.3}],
        ["label", "cv_logit", "mllm_logit", "mllm_logit_0"],
    )
    with pytest.raises(ValueError):
        load_signals_csv(mixed)


def _synthetic_signals(num_samples, seed, cv_noise=0.4, mllm_noise=0.9):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, num_samples)
    sign = 2 * labels - 1
    cv_logits = sign * rng.uniform(0.6, 1.4, num_samples) + rng.normal(0, cv_noise, num_samples)
    prompt_strength = np.array([0.7, 1.0, 0.5])
    mllm_logits = (
        sign[None, :, None] * prompt_strength[None, None, :]
    ).reshape(num_samples, 3) + rng.normal(0, mllm_noise, (num_samples, 3))
    return {
        "cv_logits": cv_logits,
        "mllm_logits": mllm_logits,
        "labels": labels.astype(np.int64),
        "sample_ids": [f"s{seed}_{i}" for i in range(num_samples)],
        "domains": rng.choice(["Cartoon", "Chart", "Photo_and_poster"], num_samples).tolist(),
        "num_prompts": 3,
        "mllm_columns": ["mllm_logit_0", "mllm_logit_1", "mllm_logit_2"],
    }


def test_full_stage_two_pipeline(tmp_path):
    train = _synthetic_signals(320, seed=1)
    dev = _synthetic_signals(160, seed=2)
    test = _synthetic_signals(320, seed=3)

    result = train_dispatcher(
        train,
        dev,
        hidden_dim=64,
        epochs=30,
        rejection_budget=0.15,
        device="cpu",
        seed=2026,
    )
    temperatures = result["temperatures"]
    assert 0.05 <= temperatures["temperature_cv"] <= 5.0
    assert 0.05 <= temperatures["temperature_mllm"] <= 5.0

    metrics = evaluate_dispatcher(
        result["dispatcher"],
        test,
        temperature_cv=temperatures["temperature_cv"],
        temperature_mllm=temperatures["temperature_mllm"],
        abstention_threshold_value=result["abstention_threshold"],
    )
    assert metrics["accuracy"] > 0.85
    assert 0.0 <= metrics["mean_gate_weight_cv"] <= 1.0
    assert 0.0 <= metrics["ap"] <= 1.0
    assert metrics["selective"]["selective_accuracy"] >= metrics["accuracy"]

    # On the validation split itself the threshold targets the 15% budget.
    dev_scored = evaluate_dispatcher(
        result["dispatcher"],
        dev,
        temperature_cv=temperatures["temperature_cv"],
        temperature_mllm=temperatures["temperature_mllm"],
        abstention_threshold_value=result["abstention_threshold"],
    )
    deferred_fraction = 1.0 - dev_scored["selective"]["coverage"]
    assert 0.05 <= deferred_fraction <= 0.30
    assert set(dev_scored["per_domain"]) == {"Cartoon", "Chart", "Photo_and_poster"}

    checkpoint_path = tmp_path / "dispatcher.pth"
    save_dispatcher_checkpoint(
        checkpoint_path,
        dispatcher=result["dispatcher"],
        config={"hidden_dim": 64},
        temperature_cv=temperatures["temperature_cv"],
        temperature_mllm=temperatures["temperature_mllm"],
        abstention_threshold_value=result["abstention_threshold"],
        num_prompts=3,
        metrics={"dev_accuracy": dev_scored["accuracy"]},
    )
    restored, payload = load_dispatcher_checkpoint(checkpoint_path)
    assert payload["temperature_cv"] == temperatures["temperature_cv"]

    restored_metrics = evaluate_dispatcher(
        restored,
        test,
        temperature_cv=payload["temperature_cv"],
        temperature_mllm=payload["temperature_mllm"],
        abstention_threshold_value=payload["abstention_threshold"],
    )
    np.testing.assert_allclose(
        metrics["rows"]["probability"],
        restored_metrics["rows"]["probability"],
        rtol=1e-6,
    )


def test_write_and_reload_signals_csv(tmp_path):
    source = _synthetic_signals(20, seed=9)
    path = tmp_path / "roundtrip.csv"
    write_signals_csv(
        path,
        cv_logits=source["cv_logits"],
        mllm_logits=source["mllm_logits"],
        labels=source["labels"],
        sample_ids=source["sample_ids"],
        domains=source["domains"],
    )
    reloaded = load_signals_csv(path)
    np.testing.assert_allclose(reloaded["cv_logits"], source["cv_logits"])
    np.testing.assert_allclose(reloaded["mllm_logits"], source["mllm_logits"])
    np.testing.assert_array_equal(reloaded["labels"], source["labels"])
    assert reloaded["num_prompts"] == 3


def test_mismatched_prompt_counts_rejected():
    train = _synthetic_signals(40, seed=1)
    dev = _synthetic_signals(40, seed=2)
    dev["mllm_logits"] = dev["mllm_logits"][:, :2]
    dev["num_prompts"] = 2
    with pytest.raises(ValueError):
        train_dispatcher(train, dev, epochs=2)
