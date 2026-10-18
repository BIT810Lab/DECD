#!/usr/bin/env python3
"""Join the two frozen experts' scalar outputs into a dispatcher signals CSV.

The dispatcher is trained solely on scalar signals, so each split only needs:

* ``cv_logit``        - Visual Expert logit per sample, taken from an
                        ``evaluate_visual_expert.py`` predictions file
                        (``predictions.csv`` or the JSON row list),
* ``mllm_logit_<k>``  - one logit per reasoning prompt from the trained MLLM
                        semantic expert evaluated on the cached frozen features.

Rows are aligned by ``sample_id``. Produce one signals CSV per split (train,
dev, eval) before running ``train_dispatcher.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from decd.dispatcher.data import write_signals_csv  # noqa: E402
from decd.mllm_expert.checkpoint import load_expert_checkpoint  # noqa: E402
from decd.mllm_expert.features import load_feature_cache  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build one aligned scalar-signals CSV for the dispatcher."
    )
    parser.add_argument(
        "--features",
        type=Path,
        required=True,
        help="Frozen MLLM feature cache for the split (*_features.pt).",
    )
    parser.add_argument(
        "--mllm-checkpoint",
        type=Path,
        required=True,
        help="Trained MLLM semantic expert checkpoint.",
    )
    parser.add_argument(
        "--cv-predictions",
        type=Path,
        required=True,
        help=(
            "Visual Expert predictions file containing sample_id and logit "
            "(predictions.csv or a JSON list of prediction rows)."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_cv_scores(path: Path) -> dict[str, float]:
    """Read sample_id -> logit from a predictions CSV or JSON list."""

    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload if isinstance(payload, list) else payload.get("predictions", [])
    else:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"no Visual Expert predictions found in {path}")
    scores: dict[str, float] = {}
    for row in rows:
        sample_id = str(row.get("sample_id") or "").strip()
        if not sample_id or "logit" not in row or row["logit"] in (None, ""):
            raise ValueError(
                f"every row needs sample_id and logit; check the format of {path}"
            )
        scores[sample_id] = float(row["logit"])
    if len(scores) != len(rows):
        raise ValueError(f"duplicate sample_id values in {path}")
    return scores


@torch.no_grad()
def mllm_prompt_logits(
    expert,
    features: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for start in range(0, features.shape[0], batch_size):
        batch = features[start : start + batch_size].to(device)
        logits = expert(batch)["prompt_logits"].float().cpu().numpy()
        chunks.append(logits)
    return np.concatenate(chunks, axis=0)


def main() -> None:
    args = parse_args()
    output = args.output.expanduser().resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"output exists; pass --overwrite: {output}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    cache = load_feature_cache(args.features)
    expert, checkpoint_payload = load_expert_checkpoint(
        args.mllm_checkpoint, device=device
    )
    expert.eval()

    sample_ids: list[str] = cache["sample_ids"]
    labels = cache["labels"].numpy().astype(np.int64)
    features = cache["features"]
    if features.shape[1] != int(checkpoint_payload["num_prompts"]):
        raise ValueError(
            f"feature cache has {features.shape[1]} prompt views but the "
            f"expert expects {checkpoint_payload['num_prompts']}"
        )

    cv_scores = load_cv_scores(args.cv_predictions.expanduser().resolve())
    missing = [sample_id for sample_id in sample_ids if sample_id not in cv_scores]
    extra = sorted(set(cv_scores).difference(sample_ids))
    if missing or extra:
        raise ValueError(
            f"sample_id mismatch: {len(missing)} missing CV scores, "
            f"{len(extra)} unmatched CV scores"
        )
    cv_logits = np.array([cv_scores[sample_id] for sample_id in sample_ids])

    mllm_logits = mllm_prompt_logits(
        expert, features, device=device, batch_size=int(args.batch_size)
    )
    if not np.all(np.isfinite(mllm_logits)) or not np.all(np.isfinite(cv_logits)):
        raise ValueError("expert logits contain NaN/Inf; check the feature cache")

    domains = cache.get("domains", ["all"] * len(sample_ids))
    write_signals_csv(
        output,
        cv_logits=cv_logits,
        mllm_logits=mllm_logits,
        labels=labels,
        sample_ids=sample_ids,
        domains=domains,
    )

    mllm_prob = 1.0 / (1.0 + np.exp(-mllm_logits.mean(axis=1)))
    cv_prob = 1.0 / (1.0 + np.exp(-cv_logits))
    print(
        f"{len(sample_ids)} samples -> {output}\n"
        f"  cv_acc   = {float(((cv_prob >= 0.5).astype(int) == labels).mean()):.4f}\n"
        f"  mllm_acc = "
        f"{float(((mllm_prob >= 0.5).astype(int) == labels).mean()):.4f}"
    )


if __name__ == "__main__":
    main()
