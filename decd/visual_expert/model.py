"""OpenCLIP detector construction, trainable parameters, and inference."""

from __future__ import annotations

import math
import os
import random
from collections.abc import Iterable
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader


def set_determinism(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        try:
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
        except AttributeError:
            pass


class CLIPBinaryDetector(nn.Module):
    """A single shared Real/Fake head on OpenCLIP image features."""

    def __init__(self, clip_model: nn.Module, feature_dim: int) -> None:
        super().__init__()
        self.clip_model = clip_model
        self.head = nn.Linear(feature_dim, 1)

    def forward(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.clip_model.encode_image(images, normalize=False)
        return self.head(features).squeeze(1), features


def infer_feature_dim(
    clip_model: nn.Module,
    image_size: int,
    device: torch.device,
) -> int:
    if hasattr(clip_model.visual, "output_dim"):
        return int(clip_model.visual.output_dim)
    with torch.no_grad():
        dummy = torch.zeros(1, 3, image_size, image_size, device=device)
        features = clip_model.encode_image(dummy, normalize=False)
    return int(features.shape[-1])


def configure_trainable_parameters(
    detector: CLIPBinaryDetector,
    trainable_blocks: int,
) -> dict[str, Any]:
    """Freeze the detector except the last N visual blocks, ln_post, and head."""

    for parameter in detector.parameters():
        parameter.requires_grad = False

    visual = detector.clip_model.visual
    try:
        blocks = visual.transformer.resblocks
    except AttributeError as exc:
        raise RuntimeError(
            "Visual Expert requires an OpenCLIP ViT with "
            "visual.transformer.resblocks"
        ) from exc

    if not 0 <= trainable_blocks <= len(blocks):
        raise ValueError(
            f"trainable_blocks must be in [0, {len(blocks)}], "
            f"got {trainable_blocks}"
        )
    for block in blocks[-trainable_blocks:] if trainable_blocks else []:
        block.requires_grad_(True)
    if trainable_blocks and hasattr(visual, "ln_post"):
        visual.ln_post.requires_grad_(True)
    detector.head.requires_grad_(True)

    backbone = [
        parameter
        for parameter in detector.clip_model.parameters()
        if parameter.requires_grad
    ]
    head = [parameter for parameter in detector.head.parameters()]
    trainable = backbone + head
    names = [
        name
        for name, parameter in detector.named_parameters()
        if parameter.requires_grad
    ]
    return {
        "backbone_parameters": backbone,
        "head_parameters": head,
        "trainable_parameters": trainable,
        "trainable_parameter_names": names,
        "trainable_parameter_count": sum(p.numel() for p in trainable),
        "trainable_backbone_parameter_count": sum(p.numel() for p in backbone),
        "trainable_head_parameter_count": sum(p.numel() for p in head),
        "total_visual_blocks": len(blocks),
        "trainable_visual_blocks": trainable_blocks,
    }


def build_detector(
    *,
    model_name: str,
    pretrained: str,
    image_size: int,
    trainable_blocks: int,
    device: torch.device,
    load_pretrained: bool = True,
    feature_dim: int | None = None,
    initialize_head: bool = True,
) -> tuple[CLIPBinaryDetector, dict[str, Any]]:
    """Build the OpenCLIP ViT detector.

    OpenCLIP is imported lazily so configuration, help, and data-chain checks
    can run without downloading model weights.
    """

    try:
        import open_clip
    except ImportError as exc:
        raise RuntimeError(
            "open-clip-torch is required to construct the detector"
        ) from exc

    clip_model = open_clip.create_model(
        model_name,
        pretrained=pretrained if load_pretrained else None,
    ).to(device)
    resolved_dim = (
        int(feature_dim)
        if feature_dim is not None
        else infer_feature_dim(clip_model, image_size, device)
    )
    detector = CLIPBinaryDetector(clip_model, resolved_dim).to(device)
    if initialize_head:
        nn.init.normal_(detector.head.weight, mean=0.0, std=0.01)
        nn.init.zeros_(detector.head.bias)
    parameter_info = configure_trainable_parameters(detector, trainable_blocks)
    parameter_info["feature_dim"] = resolved_dim
    return detector, parameter_info


def build_optimizer(
    parameter_info: dict[str, Any],
    *,
    backbone_lr: float,
    head_lr: float,
    weight_decay: float,
) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        [
            {
                "params": parameter_info["backbone_parameters"],
                "lr": float(backbone_lr),
            },
            {
                "params": parameter_info["head_parameters"],
                "lr": float(head_lr),
            },
        ],
        weight_decay=float(weight_decay),
    )


@torch.inference_mode()
def collect_predictions(
    detector: CLIPBinaryDetector,
    loader: DataLoader,
    *,
    device: torch.device,
    amp_enabled: bool,
) -> list[dict[str, Any]]:
    """Collect domain-agnostic logits without exposing local image paths."""

    detector.eval()
    rows: list[dict[str, Any]] = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            logits, _ = detector(images)
        logits_np = logits.float().cpu().numpy()
        labels_np = batch["label"].long().cpu().numpy()
        indices_np = batch["manifest_index"].long().cpu().numpy()

        for index in range(len(logits_np)):
            logit = float(logits_np[index])
            rows.append(
                {
                    "manifest_index": int(indices_np[index]),
                    "sample_id": str(batch["sample_id"][index]),
                    "sha256": str(batch["sha256"][index]),
                    "domain": str(batch["domain"][index]),
                    "label": int(labels_np[index]),
                    "logit": logit,
                    "probability": float(torch.sigmoid(torch.tensor(logit)).item()),
                }
            )
    return sorted(rows, key=lambda row: row["manifest_index"])


def _safe_metric(function: Any, *args: Any, **kwargs: Any) -> float:
    try:
        return float(function(*args, **kwargs))
    except ValueError:
        return float("nan")


def binary_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    predictions = (probabilities >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    ).ravel()
    two_classes = len(np.unique(labels)) == 2
    return {
        "n": int(labels.size),
        "real_n": int(np.sum(labels == 0)),
        "fake_n": int(np.sum(labels == 1)),
        "auc": (
            _safe_metric(roc_auc_score, labels, probabilities)
            if two_classes
            else float("nan")
        ),
        "ap": (
            _safe_metric(average_precision_score, labels, probabilities)
            if two_classes
            else float("nan")
        ),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": _safe_metric(
            balanced_accuracy_score,
            labels,
            predictions,
        ),
        "precision": _safe_metric(
            precision_score,
            labels,
            predictions,
            zero_division=0,
        ),
        "recall": _safe_metric(
            recall_score,
            labels,
            predictions,
            zero_division=0,
        ),
        "f1": _safe_metric(
            f1_score,
            labels,
            predictions,
            zero_division=0,
        ),
        "fake_recall": float(tp / (tp + fn)) if tp + fn else float("nan"),
        "real_specificity": (
            float(tn / (tn + fp)) if tn + fp else float("nan")
        ),
        "fpr": float(fp / (fp + tn)) if fp + tn else float("nan"),
        "fnr": float(fn / (fn + tp)) if fn + tp else float("nan"),
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def _finite(values: Iterable[float]) -> list[float]:
    return [float(value) for value in values if math.isfinite(float(value))]


def summarize_predictions(
    rows: list[dict[str, Any]],
    threshold: float,
) -> dict[str, Any]:
    labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
    probabilities = np.asarray(
        [row["probability"] for row in rows],
        dtype=np.float64,
    )
    overall = binary_metrics(labels, probabilities, threshold)

    per_domain: dict[str, dict[str, Any]] = {}
    domains = sorted({str(row["domain"]) for row in rows})
    for domain in domains:
        selected = [row for row in rows if row["domain"] == domain]
        domain_labels = np.asarray(
            [row["label"] for row in selected],
            dtype=np.int64,
        )
        domain_probabilities = np.asarray(
            [row["probability"] for row in selected],
            dtype=np.float64,
        )
        per_domain[domain] = binary_metrics(
            domain_labels,
            domain_probabilities,
            threshold,
        )

    domain_metrics = list(per_domain.values())
    aggregate: dict[str, float] = {}
    for name in ("auc", "ap", "balanced_accuracy", "f1"):
        values = _finite(item[name] for item in domain_metrics)
        aggregate[f"macro_{name}"] = (
            float(np.mean(values)) if values else float("nan")
        )
    for output_name, metric_name in (
        ("worst_auc", "auc"),
        ("worst_ap", "ap"),
        ("worst_fake_recall", "fake_recall"),
        ("worst_real_specificity", "real_specificity"),
    ):
        values = _finite(item[metric_name] for item in domain_metrics)
        aggregate[output_name] = (
            float(np.min(values)) if values else float("nan")
        )
    for output_name, metric_name in (
        ("fpr_gap", "fpr"),
        ("fnr_gap", "fnr"),
        ("auc_gap", "auc"),
    ):
        values = _finite(item[metric_name] for item in domain_metrics)
        aggregate[output_name] = (
            float(np.max(values) - np.min(values))
            if values
            else float("nan")
        )
    return {
        "overall": overall,
        "per_domain": per_domain,
        "aggregate": aggregate,
    }
