"""Cross-prompt gated bottleneck adapter (the PP adapter).

The trainable residual of every prompt view is scaled by a deterministic,
parameter-free consensus gate ``g``: the cosine agreement of that frozen view
with the other ``K_p - 1`` views of the same image. Gradients are stopped at
the gate, and with a zero-initialized up-projection training starts from
``Norm(z0)``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def consensus_gates(z0: torch.Tensor) -> torch.Tensor:
    """Return per-view gates ``g_i^(k) = (1 + c_i^(k)) / 2``.

    Args:
        z0: frozen prompt-conditioned representations, shape ``(..., K, d)``.

    Returns:
        Gates in ``[0, 1]``, shape ``(..., K)``. A lone view (``K == 1``)
        has no peers and receives gate ``1`` so the full residual applies.
    """

    if z0.ndim < 2:
        raise ValueError("z0 must have shape (..., K, d)")
    num_prompts = z0.shape[-2]
    if num_prompts == 1:
        return torch.ones(z0.shape[:-1], device=z0.device, dtype=z0.dtype)

    normalized = F.normalize(z0, p=2, dim=-1, eps=1.0e-8)
    similarity = torch.matmul(normalized, normalized.transpose(-2, -1))
    # Diagonal cosine similarities equal 1; average over the K-1 peers.
    agreement = (similarity.sum(dim=-1) - 1.0) / (num_prompts - 1)
    agreement = agreement.clamp(min=-1.0, max=1.0)
    return (1.0 + agreement) * 0.5


class GatedBottleneckAdapter(nn.Module):
    """``z_a = Norm(z0 + g * W2 sigma_L(W1 z0))`` with ``W2`` zero-initialized."""

    def __init__(
        self,
        feature_dim: int,
        bottleneck_ratio: int = 4,
    ) -> None:
        super().__init__()
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if bottleneck_ratio <= 0 or feature_dim % bottleneck_ratio != 0:
            raise ValueError(
                "feature_dim must be divisible by bottleneck_ratio "
                f"(feature_dim={feature_dim}, ratio={bottleneck_ratio})"
            )
        self.feature_dim = int(feature_dim)
        self.bottleneck_dim = feature_dim // bottleneck_ratio
        self.down_projection = nn.Linear(feature_dim, self.bottleneck_dim, bias=False)
        self.up_projection = nn.Linear(self.bottleneck_dim, feature_dim, bias=False)
        self.norm = nn.LayerNorm(feature_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.down_projection.weight, a=5.0**0.5)
        # Zero residual at step zero: the adapted stream starts at Norm(z0).
        nn.init.zeros_(self.up_projection.weight)

    def forward(
        self,
        z0: torch.Tensor,
        gates: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Adapt a batch of prompt-view representations.

        Args:
            z0: frozen representations, shape ``(B, K, d)``.
            gates: optional precomputed gates ``(B, K)``; computed from ``z0``
                otherwise. Gates always receive stopped gradients.

        Returns:
            Tuple ``(z_a, gates)`` with ``z_a`` of shape ``(B, K, d)``.
        """

        if gates is None:
            gates = consensus_gates(z0)
        gates = gates.detach()
        residual = self.up_projection(F.gelu(self.down_projection(z0)))
        adapted = self.norm(z0 + gates.unsqueeze(-1) * residual)
        return adapted, gates
