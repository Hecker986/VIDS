"""Sparse segment evidence pooling for CAN windows."""
from __future__ import annotations

import torch
from torch import nn


class SegmentTopKPooling(nn.Module):
    """Score local segments and pool the strongest sparse evidence."""

    def __init__(self, numeric_dim: int = 10, num_segments: int = 10, top_k: int = 2, hidden_dim: int = 64):
        super().__init__()
        self.num_segments = num_segments
        self.top_k = top_k
        self.scorer = nn.Sequential(
            nn.Linear(numeric_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, frame_numeric: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, length, dim = frame_numeric.shape
        segments = min(self.num_segments, max(int(length), 1))
        usable = (length // segments) * segments
        if usable == 0:
            segment_features = frame_numeric.mean(dim=1, keepdim=True)
        else:
            x = frame_numeric[:, :usable].reshape(bsz, segments, usable // segments, dim)
            segment_features = x.mean(dim=2)
        segment_scores = torch.sigmoid(self.scorer(segment_features)).squeeze(-1)
        k = min(self.top_k, segment_scores.shape[1])
        topk_scores, topk_idx = torch.topk(segment_scores, k=k, dim=1)
        return topk_scores.mean(dim=1), segment_scores, topk_idx
