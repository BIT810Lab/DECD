"""Stage-II training: calibrate frozen experts, then learn only the gate."""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .abstention import abstention_threshold
from .calibration import apply_temperature, fit_expert_temperatures
from .model import (
    ConfidenceAwareDispatcher,
    dispatcher_loss,
    fused_probability,
    routed_confidence,
)
from .signals import prompt_summary, reliability_vector


def set_determinism(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def prepare_inputs(
    signals: dict[str, Any],
    *,
    temperature_cv: float,
    temperature_mllm: float,
) -> dict[str, Any]:
    """Calibrate logits and build the six-dimensional reliability vectors."""

    labels = np.asarray(signals["labels"], dtype=np.int64)
    cv_probabilities = apply_temperature(signals["cv_logits"], temperature_cv)
    mllm_prompt_probabilities = apply_temperature(
        signals["mllm_logits"],
        temperature_mllm,
    )
    mean_probability, variance = prompt_summary(mllm_prompt_probabilities)
    vectors = reliability_vector(
        cv_probabilities,
        mean_probability,
        variance,
    )
    return {
        "reliability": torch.tensor(vectors, dtype=torch.float32),
        "p_cv": torch.tensor(cv_probabilities, dtype=torch.float32),
        "p_mllm": torch.tensor(mean_probability, dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.float32),
    }


def _tensors_to_device(batch: tuple[torch.Tensor, ...], device: torch.device):
    return tuple(tensor.to(device) for tensor in batch)


@torch.no_grad()
def _dev_loss(
    dispatcher: ConfidenceAwareDispatcher,
    dev_inputs: dict[str, torch.Tensor],
    device: torch.device,
) -> float:
    dispatcher.eval()
    logits = dispatcher(dev_inputs["reliability"].to(device))
    loss = dispatcher_loss(
        logits,
        dev_inputs["p_cv"].to(device),
        dev_inputs["p_mllm"].to(device),
        dev_inputs["labels"].to(device),
    )
    return float(loss.item())


def train_dispatcher(
    train_signals: dict[str, Any],
    dev_signals: dict[str, Any],
    *,
    hidden_dim: int = 64,
    learning_rate: float = 1.0e-3,
    weight_decay: float = 1.0e-4,
    batch_size: int = 32,
    epochs: int = 20,
    early_stopping_patience: int = 5,
    rejection_budget: float = 0.15,
    device: torch.device | str = "cpu",
    seed: int = 2026,
    fixed_temperature_cv: float | None = None,
    fixed_temperature_mllm: float | None = None,
) -> dict[str, Any]:
    """Fit temperatures on the validation split, then train only the MLP."""

    if train_signals["num_prompts"] != dev_signals["num_prompts"]:
        raise ValueError("train and dev must use the same number of MLLM prompts")
    set_determinism(seed)
    device = torch.device(device)
    temperatures = fit_expert_temperatures(
        cv_logits=dev_signals["cv_logits"],
        mllm_logits=dev_signals["mllm_logits"],
        labels=dev_signals["labels"],
        cv_temperature=fixed_temperature_cv,
        mllm_temperature=fixed_temperature_mllm,
    )

    train_inputs = prepare_inputs(
        train_signals,
        temperature_cv=temperatures["temperature_cv"],
        temperature_mllm=temperatures["temperature_mllm"],
    )
    dev_inputs = prepare_inputs(
        dev_signals,
        temperature_cv=temperatures["temperature_cv"],
        temperature_mllm=temperatures["temperature_mllm"],
    )

    loader = DataLoader(
        TensorDataset(
            train_inputs["reliability"],
            train_inputs["p_cv"],
            train_inputs["p_mllm"],
            train_inputs["labels"],
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    dispatcher = ConfidenceAwareDispatcher(hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.AdamW(
        dispatcher.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )

    best_loss = float("inf")
    best_state = None
    stale_epochs = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        dispatcher.train()
        train_losses: list[float] = []
        for reliability, p_cv, p_mllm, labels in loader:
            reliability, p_cv, p_mllm, labels = _tensors_to_device(
                (reliability, p_cv, p_mllm, labels),
                device,
            )
            optimizer.zero_grad(set_to_none=True)
            loss = dispatcher_loss(
                dispatcher(reliability),
                p_cv,
                p_mllm,
                labels,
            )
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        dev_loss = _dev_loss(dispatcher, dev_inputs, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(train_losses)),
                "dev_loss": dev_loss,
            }
        )
        if dev_loss < best_loss - 1.0e-7:
            best_loss = dev_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in dispatcher.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        if early_stopping_patience and stale_epochs >= early_stopping_patience:
            break

    if best_state is not None:
        dispatcher.load_state_dict(best_state)

    tau_r = validation_abstention_threshold(
        dispatcher,
        dev_inputs,
        device=device,
        rejection_budget=rejection_budget,
    )
    return {
        "dispatcher": dispatcher,
        "temperatures": temperatures,
        "abstention_threshold": tau_r,
        "history": history,
        "best_dev_loss": best_loss,
        "dev_inputs": dev_inputs,
    }


@torch.no_grad()
def validation_abstention_threshold(
    dispatcher: ConfidenceAwareDispatcher,
    dev_inputs: dict[str, torch.Tensor],
    *,
    device: torch.device | str,
    rejection_budget: float,
) -> float:
    dispatcher.eval()
    gate_logits = dispatcher(dev_inputs["reliability"].to(device))
    fused, weights = fused_probability(
        gate_logits,
        dev_inputs["p_cv"].to(device),
        dev_inputs["p_mllm"].to(device),
    )
    confidence = routed_confidence(weights, fused).float().cpu().numpy()
    return abstention_threshold(confidence, rejection_budget)
