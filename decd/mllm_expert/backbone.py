"""Frozen MLLM backbone producing pooled prompt-conditioned hidden states.

The paper uses a completely frozen Qwen2.5-VL-7B-Instruct. ``transformers`` is
imported lazily so that adapter code and configuration checks run without
the 7B checkpoint (mirroring the Visual Expert's lazy
OpenCLIP import). Any object exposing ``feature_dim`` and ``encode(images,
prompts) -> Tensor[B, K, d]`` can be used as a backbone.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

import torch
from PIL import Image


@runtime_checkable
class FrozenMLLMBackbone(Protocol):
    feature_dim: int

    def encode(
        self,
        images: Sequence[Image.Image],
        prompts: Sequence[str],
    ) -> torch.Tensor: ...


def mean_pool_last_hidden(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Mean-pool non-padding tokens of the last hidden layer."""

    # Accumulate in float32: summing hundreds of fp16 hidden-state tokens
    # easily overflows the fp16 range and yields NaN pooled features.
    hidden_states = hidden_states.float()
    mask = attention_mask.to(torch.float32).unsqueeze(-1)
    summed = (hidden_states * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp_min(1.0)
    return summed / counts


class Qwen25VLBackbone:
    """Frozen Qwen2.5-VL backbone with a mean-pooled last-layer readout."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        *,
        device: torch.device | str = "cuda:0",
        # bfloat16 is the safe default: fp16 summation inside the Qwen2.5-VL
        # vision tower (and token mean-pooling) overflows on some inputs,
        # producing NaN features. bfloat16 has the same memory footprint and
        # is natively supported by every Ampere-class (or newer) GPU.
        dtype: torch.dtype = torch.bfloat16,
        pooling: str = "mean",
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 512 * 28 * 28,
        model_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if pooling not in {"mean", "last"}:
            raise ValueError("pooling must be 'mean' or 'last'")
        try:
            import transformers
        except ImportError as exc:
            raise RuntimeError(
                "transformers>=4.49 is required for the Qwen2.5-VL backbone"
            ) from exc
        if not hasattr(transformers, "Qwen2_5_VLForConditionalGeneration"):
            raise RuntimeError(
                "the installed transformers does not provide "
                "Qwen2_5_VLForConditionalGeneration; install transformers>=4.49"
            )

        self.model_name = model_name
        self.device = torch.device(device)
        self.pooling = pooling
        self.min_pixels = int(min_pixels)
        self.max_pixels = int(max_pixels)
        self.processor = transformers.AutoProcessor.from_pretrained(model_name)
        self.model = transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
            **(model_kwargs or {}),
        ).to(self.device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.feature_dim = int(self.model.config.hidden_size)

    def _conversation_text(self, prompt: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        return self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    @torch.no_grad()
    def encode(
        self,
        images: Sequence[Image.Image],
        prompts: Sequence[str],
    ) -> torch.Tensor:
        """Return pooled frozen representations of shape ``(B, K, d)``."""

        if not images or not prompts:
            raise ValueError("both images and prompts must be non-empty")
        batch_size = len(images)
        num_prompts = len(prompts)

        flat_images: list[Image.Image] = []
        flat_texts: list[str] = []
        for image in images:
            prepared = image.convert("RGB")
            for prompt in prompts:
                flat_images.append(prepared)
                flat_texts.append(self._conversation_text(prompt))

        processor_kwargs: dict[str, Any] = {
            "text": flat_texts,
            "images": flat_images,
            "padding": True,
            "return_tensors": "pt",
        }
        try:
            inputs = self.processor(
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
                **processor_kwargs,
            ).to(self.device)
        except TypeError:
            inputs = self.processor(**processor_kwargs).to(self.device)
        outputs = self.model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[-1]
        if self.pooling == "mean":
            pooled = mean_pool_last_hidden(hidden, inputs["attention_mask"])
        else:
            row_lengths = inputs["attention_mask"].sum(dim=1) - 1
            pooled = hidden[torch.arange(hidden.shape[0], device=hidden.device), row_lengths]
        return pooled.to(torch.float32).reshape(batch_size, num_prompts, -1).cpu()
