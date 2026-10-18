"""End-to-end Stage-I training tests on synthetic cached frozen features."""

import torch

from decd.mllm_expert.checkpoint import (
    load_expert_checkpoint,
    save_expert_checkpoint,
)
from decd.mllm_expert.features import build_feature_loader
from decd.mllm_expert.model import build_optimizer, build_semantic_expert
from decd.mllm_expert.prompts import REASONING_PROMPTS
from decd.mllm_expert.training import (
    evaluate_expert,
    set_determinism,
    train_semantic_expert,
)

FEATURE_DIM = 32
NUM_PROMPTS = 3


def _shared_geometry():
    """Fixed label direction and prompt offsets shared across all splits."""

    generator = torch.Generator().manual_seed(1234)
    class_direction = torch.randn(FEATURE_DIM, generator=generator)
    class_direction /= class_direction.norm()
    prompt_offsets = 0.3 * torch.randn(
        NUM_PROMPTS,
        FEATURE_DIM,
        generator=generator,
    )
    return class_direction, prompt_offsets


CLASS_DIRECTION, PROMPT_OFFSETS = _shared_geometry()


def _synthetic_cache(num_samples, seed):
    """Frozen features that linearly encode the label plus prompt offsets."""

    generator = torch.Generator().manual_seed(seed)
    labels = torch.randint(0, 2, (num_samples,), generator=generator).float()
    noise = 0.25 * torch.randn(
        num_samples,
        NUM_PROMPTS,
        FEATURE_DIM,
        generator=generator,
    )
    features = (
        (2 * labels - 1).view(-1, 1, 1) * CLASS_DIRECTION.view(1, 1, -1)
        + PROMPT_OFFSETS.view(1, NUM_PROMPTS, -1)
        + noise
    )
    return {
        "features": features,
        "labels": labels,
        "sample_ids": [f"s{seed}_{i}" for i in range(num_samples)],
        "domains": ["all"] * num_samples,
        "prompts": list(REASONING_PROMPTS),
        "feature_dim": FEATURE_DIM,
    }


def test_training_improves_and_selects_best_epoch():
    set_determinism(2026)
    train_cache = _synthetic_cache(128, seed=1)
    dev_cache = _synthetic_cache(64, seed=2)
    expert, info = build_semantic_expert(
        feature_dim=FEATURE_DIM,
        num_prompts=NUM_PROMPTS,
        bottleneck_ratio=4,
        device="cpu",
        prompts=REASONING_PROMPTS,
    )
    optimizer = build_optimizer(expert, learning_rate=1.0e-2, weight_decay=1.0e-5)
    train_loader = build_feature_loader(train_cache, batch_size=16, shuffle=True)
    dev_loader = build_feature_loader(dev_cache, batch_size=32, shuffle=False)

    before = evaluate_expert(expert, dev_loader, device=torch.device("cpu"))
    result = train_semantic_expert(
        expert,
        train_loader,
        dev_loader,
        optimizer=optimizer,
        epochs=40,
        np_temperature=0.1,
        np_lambda=0.5,
        device="cpu",
    )
    after = evaluate_expert(expert, dev_loader, device=torch.device("cpu"))

    assert after["macro_f1"] > before["macro_f1"]
    assert after["macro_f1"] >= 0.8
    assert result["best_epoch"] >= 1
    # Loss decreased during training.
    losses = [row["train_loss"] for row in result["history"]]
    assert losses[-1] < losses[0]
    # Trainable parameter count stays small: adapter + head only.
    assert info["trainable_parameter_count"] == sum(
        p.numel() for p in expert.parameters()
    )


def test_checkpoint_roundtrip_preserves_predictions(tmp_path):
    set_determinism(7)
    train_cache = _synthetic_cache(64, seed=3)
    dev_cache = _synthetic_cache(32, seed=4)
    expert, _ = build_semantic_expert(
        feature_dim=FEATURE_DIM,
        num_prompts=NUM_PROMPTS,
        bottleneck_ratio=4,
        device="cpu",
        prompts=REASONING_PROMPTS,
    )
    optimizer = build_optimizer(expert, learning_rate=1.0e-2, weight_decay=1.0e-5)
    result = train_semantic_expert(
        expert,
        build_feature_loader(train_cache, batch_size=16, shuffle=True),
        build_feature_loader(dev_cache, batch_size=32, shuffle=False),
        optimizer=optimizer,
        epochs=10,
        device="cpu",
    )

    checkpoint_path = tmp_path / "mllm_expert.pth"
    save_expert_checkpoint(
        checkpoint_path,
        expert=expert,
        config={"np_temperature": 0.1, "np_lambda": 0.5},
        prompts=REASONING_PROMPTS,
        metadata={"best_epoch": result["best_epoch"]},
    )
    restored, payload = load_expert_checkpoint(checkpoint_path, device="cpu")
    assert payload["feature_dim"] == FEATURE_DIM
    assert payload["num_prompts"] == NUM_PROMPTS

    expert.eval()
    restored.eval()
    with torch.no_grad():
        expected = expert(dev_cache["features"])["prompt_logits"]
        actual = restored(dev_cache["features"])["prompt_logits"]
    assert torch.allclose(expected, actual, atol=1e-6)
