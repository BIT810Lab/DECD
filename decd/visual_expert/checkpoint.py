"""Checkpoint save, restore, configuration, and path sanitization."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Mapping
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import torch
import yaml

from .model import build_detector, build_optimizer
from .losses import require_finite

CONTRACT_KEYS = ('model_name', 'pretrained', 'image_size', 'trainable_blocks',
                 'seed', 'batch_size', 'backbone_lr', 'head_lr', 'weight_decay',
                 'amp', 'reference_batches')


def validate_training_config(config):
    for key in CONTRACT_KEYS + ('repair_batches', 'gamma', 'repair_lambda',
            'ramp_batches', 'rho_max', 'gradient_epsilon', 'repair_label_side',
            'risk_column', 'risk_normalization', 'risk_normalization_alpha'):
        if key not in config:
            raise ValueError(f'missing configuration field: {key}')
    if (config['model_name'], config['pretrained'], config['image_size'],
            config['trainable_blocks']) != ('ViT-L-14-quickgelu', 'openai', 224, 2):
        raise ValueError('this extracted protocol requires OpenAI ViT-L/14, 224, last two blocks')
    for key in ('seed', 'batch_size', 'reference_batches', 'repair_batches', 'ramp_batches'):
        if isinstance(config[key], bool) or not isinstance(config[key], int) or config[key] < 0:
            raise ValueError(f'invalid integer configuration: {key}')
    if config['batch_size'] <= 0 or config['batch_size'] % 6:
        raise ValueError('batch_size must be a positive multiple of six')
    if config['reference_batches'] != 100 or config['repair_batches'] != 50:
        raise ValueError('the extracted endpoint protocol is B100 -> B50')
    for key in ('backbone_lr', 'head_lr', 'weight_decay', 'gamma', 'repair_lambda',
                'rho_max', 'gradient_epsilon', 'risk_normalization_alpha'):
        if not math.isfinite(float(config[key])) or float(config[key]) < 0:
            raise ValueError(f'invalid finite configuration: {key}')
    if any(float(config[k]) <= 0 for k in ('backbone_lr', 'head_lr', 'rho_max',
                                          'gradient_epsilon', 'risk_normalization_alpha')):
        raise ValueError('learning rates and scaling constants must be positive')
    if not isinstance(config['amp'], bool) or config['repair_label_side'] != 'real':
        raise ValueError('invalid AMP setting or non-real repair side')
    # Validate the existing configurable alternatives, without changing the
    # reported YAML (mean hinge, gamma0.5, minmax). Do not impose a new loss.
    margin = f"m{float(config['gamma']):g}".replace('.', 'p')
    if config['risk_column'] not in (f'risk_mean_{margin}', f'risk_sq_mean_{margin}'):
        raise ValueError('risk_column does not match the configured gamma')
    if config['risk_normalization'] not in ('minmax', 'maxnorm', 'raw'):
        raise ValueError('unknown existing risk normalization')


def validate_reference_checkpoint(payload, config, train_manifest_sha256, train_content_sha256):
    meta = payload.get('metadata', {})
    if meta.get('stage') != 'reference' or meta.get('completed_batches') != config['reference_batches']:
        raise ValueError('parent must be a completed B100 reference endpoint')
    for key in CONTRACT_KEYS:
        if payload.get('config', {}).get(key) != config.get(key):
            raise ValueError(f'reference configuration mismatch: {key}')
    for key, value in [('train_manifest_sha256', train_manifest_sha256),
                       ('train_content_sha256', train_content_sha256)]:
        if not value or meta.get(key) != value:
            raise ValueError(f'reference data provenance mismatch: {key}')
    state = payload.get('optimizer_state_dict')
    if not isinstance(state, dict) or not state.get('state'):
        raise ValueError('B100 AdamW state is required')
    groups = state.get('param_groups', [])
    if len(groups) != 2:
        raise ValueError('expected backbone/head AdamW groups')
    for group, lr in zip(groups, [config['backbone_lr'], config['head_lr']]):
        if group.get('lr') != lr or group.get('weight_decay') != config['weight_decay']:
            raise ValueError('AdamW learning rate / weight decay provenance mismatch')
        if tuple(group.get('betas', ())) != (.9, .999) or group.get('eps') != 1e-8:
            raise ValueError('unexpected AdamW mathematics')
    for item in state['state'].values():
        step = item.get('step', -1)
        step = float(step.item()) if torch.is_tensor(step) else float(step)
        if step != config['reference_batches']:
            raise ValueError('AdamW step is not the B100 endpoint')

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
    for value in detector.state_dict().values():
        require_finite(value, 'saved model state')
    if optimizer is not None:
        for item in optimizer.state.values():
            for value in item.values():
                if torch.is_tensor(value):
                    require_finite(value, 'saved AdamW state')
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
        weights_only=True,
    )
    if not isinstance(payload, dict):
        raise ValueError("invalid checkpoint payload")
    if payload.get('format_version') != 1:
        raise ValueError('unsupported checkpoint format; no implicit legacy conversion')
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
    expected_config: dict | None = None,
    train_manifest_sha256: str | None = None,
    train_content_sha256: str | None = None,
    require_reference: bool = False,
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
        raise ValueError('checkpoint configuration must be a mapping')
    for key in ('model_name', 'pretrained', 'image_size', 'trainable_blocks'):
        if key not in config:
            raise ValueError(f'checkpoint missing model configuration: {key}')
    if require_reference:
        if expected_config is None:
            raise ValueError('reference provenance requires an active configuration')
        validate_reference_checkpoint(payload, expected_config, train_manifest_sha256, train_content_sha256)

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
    for key, value in state_dict.items():
        if not torch.is_tensor(value):
            raise ValueError(f'non-tensor model state: {key}')
        require_finite(value, 'checkpoint parameter')
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
        for group in optimizer.param_groups:
            for parameter in group['params']:
                item = optimizer.state.get(parameter, {})
                if not item:
                    raise ValueError('missing AdamW parameter state')
                for key in ('exp_avg', 'exp_avg_sq'):
                    if key not in item or item[key].shape != parameter.shape:
                        raise ValueError('AdamW state/parameter shape mismatch')
                    require_finite(item[key], 'AdamW state')
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
