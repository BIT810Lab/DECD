"""Fixed reasoning prompts (K_p = 3), identical at training and inference.

Each prompt targets one complementary facet of authenticity described in the
method section: physical lighting/shadow plausibility, geometric and
inter-object consistency, and fine texture/edge/text rendering quality. The
prompt set is fixed on purpose so per-prompt outputs of an image are directly
comparable and their cross-prompt dispersion is a meaningful instability
signal.
"""

from __future__ import annotations

REASONING_PROMPTS: tuple[str, ...] = (
    "Examine the lighting and shadows in this image. Are the light sources, "
    "illumination directions, shadow shapes, shading gradients, and their "
    "physical interactions consistent and physically plausible? Point out "
    "any physically impossible or contradictory lighting.",
    "Inspect the geometry, perspective, reflections, and relationships "
    "between objects in this image. Are spatial proportions, occlusions, "
    "symmetries, reflections, and inter-object interactions geometrically "
    "and logically consistent with a real scene?",
    "Inspect the fine details of this image: local textures, edges, and any "
    "rendered text, digits, or symbols. Are textures sharp and coherent, "
    "object boundaries natural, and glyphs correctly formed without "
    "distortion, blending artifacts, or implausible repetitions?",
)


def validate_prompts(prompts: tuple[str, ...] | list[str]) -> None:
    if not prompts:
        raise ValueError("at least one reasoning prompt is required")
    if len(set(prompts)) != len(prompts):
        raise ValueError("reasoning prompts must be distinct")
    for index, prompt in enumerate(prompts):
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"prompt {index} is empty")
