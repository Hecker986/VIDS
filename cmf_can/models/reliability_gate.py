"""Reliability-aware modality gating for Reliable-CMF-CAN."""
from __future__ import annotations

import torch
from torch import nn


class ReliabilityGate(nn.Module):
    """Estimate per-modality reliability and normalized fusion weights.

    The module consumes modality tokens plus optional shift/uncertainty
    features. It outputs reliability values in [0, 1] and normalized weights.
    """

    def __init__(self, d_model: int = 256, hidden_dim: int = 128, shift_dim: int = 7):
        super().__init__()
        self.shift_dim = shift_dim
        in_dim = d_model * 3 + shift_dim
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, tokens: torch.Tensor, shift_features: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if shift_features is None:
            shift_features = tokens.new_zeros((tokens.shape[0], self.shift_dim))
        x = torch.cat([tokens.flatten(1), shift_features.float()], dim=-1)
        reliability = torch.sigmoid(self.net(x))
        weights = reliability / reliability.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return reliability, weights


def gate_entropy(weights: torch.Tensor) -> torch.Tensor:
    """Return mean normalized entropy of a [B, M] gate distribution."""

    probs = weights.clamp_min(1e-8)
    entropy = -(probs * probs.log()).sum(dim=-1)
    return entropy.mean()
