"""Stage-I training loop for the prior-preserved semantic expert.

The loop operates on cached frozen features, so the MLLM backbone is absent
and the expert is trainable on CPU. The best epoch is selected by validation
macro-F1, matching the reported protocol.
"""

from __future__ import annotations

import copy
import os
import random
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from .losses import semantic_expert_loss
from .model import MLLMSemanticExpert, ensemble_summary


def set_determinism(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.inference_mode()
def evaluate_expert(
    expert: MLLMSemanticExpert,
    loader: DataLoader,
    *,
    device: torch.device,
    threshold: float = 0.5,
) -> dict[str, Any]:
    expert.eval()
    prompt_logits: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    gates: list[torch.Tensor] = []
    for frozen_features, labels in loader:
        outputs = expert(frozen_features.to(device))
        prompt_logits.append(outputs["prompt_logits"].float().cpu())
        gates.append(outputs["gates"].float().cpu())
        targets.append(labels.long().cpu())

    logits = torch.cat(prompt_logits, dim=0)
    labels = torch.cat(targets, dim=0).numpy()
    gate_values = torch.cat(gates, dim=0)
    mean_probability, variance = ensemble_summary(torch.sigmoid(logits))
    predictions = (mean_probability >= threshold).long().numpy()

    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "mean_prompt_variance": float(variance.mean().item()),
        "mean_gate": float(gate_values.mean().item()),
        "n": int(labels.shape[0]),
        "prompt_logits": logits,
        "labels": torch.from_numpy(labels),
    }


def train_semantic_expert(
    expert: MLLMSemanticExpert,
    train_loader: DataLoader,
    dev_loader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    np_temperature: float = 0.1,
    np_lambda: float = 0.5,
    device: torch.device | str = "cpu",
    selection_metric: str = "macro_f1",
    early_stopping_patience: int | None = None,
) -> dict[str, Any]:
    """Train adapter and head; restore the best-by-validation-metric weights."""

    if epochs <= 0:
        raise ValueError("epochs must be positive")
    device = torch.device(device)
    expert.to(device)

    best_state = copy.deepcopy(expert.state_dict())
    best_metric = -float("inf")
    best_epoch = 0
    history: list[dict[str, float]] = []
    stale_epochs = 0

    for epoch in range(1, epochs + 1):
        expert.train()
        batch_rows: list[dict[str, float]] = []
        for frozen_features, labels in train_loader:
            frozen_features = frozen_features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = expert(frozen_features)
            loss, diagnostics = semantic_expert_loss(
                prompt_logits=outputs["prompt_logits"],
                labels=labels,
                frozen=outputs["frozen"],
                adapted=outputs["adapted"],
                np_temperature=np_temperature,
                np_lambda=np_lambda,
            )
            loss.backward()
            optimizer.step()
            batch_rows.append(diagnostics)

        mean_loss = float(np.mean([row["total_loss"] for row in batch_rows]))
        evaluation = evaluate_expert(expert, dev_loader, device=device)
        metric_value = float(evaluation[selection_metric])
        history.append(
            {
                "epoch": epoch,
                "train_loss": mean_loss,
                "dev_accuracy": evaluation["accuracy"],
                f"dev_{selection_metric}": metric_value,
                "dev_mean_prompt_variance": evaluation["mean_prompt_variance"],
            }
        )
        if metric_value > best_metric:
            best_metric = metric_value
            best_epoch = epoch
            best_state = copy.deepcopy(expert.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if early_stopping_patience and stale_epochs >= early_stopping_patience:
            break

    expert.load_state_dict(best_state)
    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "selection_metric": selection_metric,
    }
