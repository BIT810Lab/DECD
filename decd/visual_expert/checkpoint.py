"""Checkpoint save, restore, configuration, and path sanitization."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import torch
import yaml

from .model import build_detector, build_optimizer

WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[\\/]")


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError("configuration root must be a mapping")
    return payload


def project_root_from_config(path: str | Path) -> Path:
    config_path = Path(path).expanduser().resolve()
    return config_path.parent.parent


def resolve_project_path(config_path: str | Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (project_root_from_config(config_path) / path).resolve()


def _is_absolute_path_text(value: str) -> bool:
    text = value.strip()
    return bool(
        WINDOWS_ABSOLUTE.match(text)
        or PureWindowsPath(text).is_absolute()
        or PurePosixPath(text).is_absolute()
    )


def _safe_path_text(value: str) -> str:
    text = value.strip()
    if WINDOWS_ABSOLUTE.match(text) or PureWindowsPath(text).is_absolute():
        return PureWindowsPath(text).name
    if PurePosixPath(text).is_absolute():
        return PurePosixPath(text).name
    return value


def sanitize_metadata(value: Any) -> Any:
    """Recursively remove absolute local paths before serialization."""

    if isinstance(value, Path):
        return value.name if value.is_absolute() else value.as_posix()
    if isinstance(value, Mapping):
        return {
            str(key): sanitize_metadata(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [sanitize_metadata(item) for item in value]
    if isinstance(value, str):
        return _safe_path_text(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def find_absolute_path_strings(
    value: Any,
    *,
    location: str = "$",
) -> list[tuple[str, str]]:
    """Recursively locate Windows, UNC, and POSIX absolute path strings."""

    findings: list[tuple[str, str]] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_location = f"{location}.<key>"
            if isinstance(key, str) and _is_absolute_path_text(key):
                findings.append((key_location, key))
            findings.extend(
                find_absolute_path_strings(
                    item,
                    location=f"{location}.{key}",
                )
            )
    elif isinstance(value, (list, tuple, set)):
        for index, item in enumerate(value):
            findings.extend(
                find_absolute_path_strings(
                    item,
                    location=f"{location}[{index}]",
                )
            )
    elif isinstance(value, str) and _is_absolute_path_text(value):
        findings.append((location, value))
    return findings


def write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            sanitize_metadata(payload),
            ensure_ascii=False,
            indent=2,
            allow_nan=True,
        ),
        encoding="utf-8",
    )


def save_checkpoint(
    path: str | Path,
    *,
    detector: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    config: dict[str, Any],
    feature_dim: int,
    metadata: dict[str, Any] | None = None,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "model_state_dict": detector.state_dict(),
        "optimizer_state_dict": (
            optimizer.state_dict() if optimizer is not None else None
        ),
        "config": sanitize_metadata(config),
        "feature_dim": int(feature_dim),
        "metadata": sanitize_metadata(metadata or {}),
    }
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, output)


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(payload, dict):
        raise ValueError("invalid checkpoint payload")
    return payload


def move_optimizer_state(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def restore_detector(
    path: str | Path,
    *,
    device: torch.device,
    restore_optimizer: bool,
) -> tuple[
    torch.nn.Module,
    torch.optim.Optimizer | None,
    dict[str, Any],
    dict[str, Any],
]:
    checkpoint_path = Path(path).expanduser().resolve()
    payload = load_checkpoint(checkpoint_path)
    config = payload.get("config", {})
    if not isinstance(config, dict):
        config = {}

    model_name = str(config.get("model_name", "ViT-L-14-quickgelu"))
    pretrained = str(config.get("pretrained", "openai"))
    image_size = int(config.get("image_size", 224))
    trainable_blocks = int(config.get("trainable_blocks", 2))
    feature_dim = int(payload.get("feature_dim", 768))

    detector, parameter_info = build_detector(
        model_name=model_name,
        pretrained=pretrained,
        image_size=image_size,
        trainable_blocks=trainable_blocks,
        device=device,
        load_pretrained=False,
        feature_dim=feature_dim,
        initialize_head=False,
    )
    state_dict = payload.get("model_state_dict", payload.get("state_dict"))
    if not isinstance(state_dict, dict):
        raise ValueError("checkpoint has no model state dictionary")
    detector.load_state_dict(state_dict, strict=True)

    optimizer = None
    if restore_optimizer:
        optimizer = build_optimizer(
            parameter_info,
            backbone_lr=float(config.get("backbone_lr", 1.0e-5)),
            head_lr=float(config.get("head_lr", 1.0e-4)),
            weight_decay=float(config.get("weight_decay", 1.0e-4)),
        )
        optimizer_state = payload.get("optimizer_state_dict")
        if not isinstance(optimizer_state, dict):
            raise ValueError(
                "reference checkpoint has no optimizer state; continuation "
                "requires the B100 optimizer state"
            )
        optimizer.load_state_dict(optimizer_state)
        move_optimizer_state(optimizer, device)

    metadata = {
        "checkpoint_name": checkpoint_path.name,
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "model_name": model_name,
        "pretrained": pretrained,
        "image_size": image_size,
        "trainable_blocks": trainable_blocks,
        "feature_dim": feature_dim,
        "optimizer_restored": bool(restore_optimizer),
    }
    return detector, optimizer, parameter_info, metadata


def export_sanitized_checkpoint(source: str | Path, target: str | Path) -> None:
    """Export a checkpoint whose metadata cannot reveal local absolute paths."""

    payload = load_checkpoint(source)
    for key in ("config", "metadata", "summary", "risk_profile"):
        if key in payload:
            payload[key] = sanitize_metadata(payload[key])

    findings = find_absolute_path_strings(payload)
    if findings:
        locations = ", ".join(location for location, _ in findings[:10])
        raise RuntimeError(
            "checkpoint still contains absolute path strings at: "
            f"{locations}"
        )

    output = Path(target)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    try:
        torch.save(payload, temporary)
        exported = load_checkpoint(temporary)
        findings = find_absolute_path_strings(exported)
        if findings:
            locations = ", ".join(location for location, _ in findings[:10])
            raise RuntimeError(
                "exported checkpoint still contains absolute path strings at: "
                f"{locations}"
            )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
