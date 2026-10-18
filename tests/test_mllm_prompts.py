"""Tests for the fixed reasoning-prompt set."""

from decd.mllm_expert.prompts import REASONING_PROMPTS, validate_prompts

import pytest


def test_reported_prompt_set():
    assert len(REASONING_PROMPTS) == 3
    assert all(isinstance(prompt, str) and prompt.strip() for prompt in REASONING_PROMPTS)
    assert len(set(REASONING_PROMPTS)) == 3
    # Every prompt stays well below the reported 256-token input budget.
    for prompt in REASONING_PROMPTS:
        assert len(prompt.split()) < 120


def test_prompt_validation():
    validate_prompts(REASONING_PROMPTS)
    with pytest.raises(ValueError):
        validate_prompts(())
    with pytest.raises(ValueError):
        validate_prompts(["same", "same"])
    with pytest.raises(ValueError):
        validate_prompts(["", "ok"])
