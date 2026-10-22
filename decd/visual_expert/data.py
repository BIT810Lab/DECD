"""Validated manifests and the unchanged letterbox/CLIP data pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset

from .manifests import DOMAIN_TO_ID, read_manifest, assert_role
from .sampler import DomainLabelBalancedBatchSampler

CLIP_MEAN = (0.48145466, 0.45782750, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
ID_TO_DOMAIN = {value: key for key, value in DOMAIN_TO_ID.items()}


def load_manifest(path, *, require_domain=True, path_field=None,
                  domain_field=None, check_files=True, role=None):
    return read_manifest(path, require_domain=require_domain, path_field=path_field,
                         domain_field=domain_field, check_files=check_files, role=role)


class LetterboxCLIPTransform:
    """Original bicubic fit, gray127 padding and CLIP normalization."""
    def __init__(self, size: int = 224) -> None:
        if size <= 0:
            raise ValueError("image size must be positive")
        self.size = int(size)
        self.mean = torch.tensor(CLIP_MEAN, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(CLIP_STD, dtype=torch.float32).view(3, 1, 1)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = ImageOps.exif_transpose(image).convert("RGB")
        width, height = image.size
        if width <= 0 or height <= 0:
            raise ValueError(f"invalid image size: {image.size}")
        scale = min(self.size / width, self.size / height)
        new_width = max(1, round(width * scale))
        new_height = max(1, round(height * scale))
        image = image.resize((new_width, new_height), Image.Resampling.BICUBIC)
        canvas = Image.new("RGB", (self.size, self.size), color=(127, 127, 127))
        canvas.paste(image, ((self.size - new_width) // 2, (self.size - new_height) // 2))
        array = np.asarray(canvas, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        return (tensor - self.mean) / self.std


class ManifestDataset(Dataset[dict[str, Any]]):
    def __init__(self, rows, image_size=224):
        self.rows = rows
        self.transform = LetterboxCLIPTransform(image_size)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        try:
            with Image.open(Path(row['image_path'])) as opened:
                image = self.transform(opened)
        except Exception as exc:
            raise RuntimeError(f"failed to read image at manifest row {index}") from exc
        if not torch.isfinite(image).all():
            raise ValueError(f"non-finite image tensor at manifest row {index}")
        return {'image': image, 'label': torch.tensor(row['label'], dtype=torch.float32),
                'domain_id': torch.tensor(row['domain_id'], dtype=torch.long),
                'domain': row['domain'], 'sample_id': row['sample_id'],
                'manifest_index': torch.tensor(row['manifest_index'], dtype=torch.long),
                'sha256': row['sha256']}


def build_training_loader(rows, *, image_size, batch_size, num_batches, seed,
                          num_workers, pin_memory, start_batch=0):
    assert_role(rows, 'train')
    sampler = DomainLabelBalancedBatchSampler(rows, batch_size=batch_size,
                                              num_batches=num_batches, seed=seed,
                                              start_batch=start_batch)
    return DataLoader(ManifestDataset(rows, image_size), batch_sampler=sampler,
                      num_workers=num_workers, pin_memory=pin_memory,
                      persistent_workers=num_workers > 0)


def build_eval_loader(rows, *, image_size, batch_size, num_workers, pin_memory):
    return DataLoader(ManifestDataset(rows, image_size), batch_size=batch_size,
                      shuffle=False, num_workers=num_workers, pin_memory=pin_memory,
                      persistent_workers=num_workers > 0)
