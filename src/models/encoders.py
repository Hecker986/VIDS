"""Model encoders for SCS-CAN."""
from __future__ import annotations

import torch
from torch import nn


class PayloadEncoder(nn.Module):
    def __init__(self, d_byte=16, d_payload=128):
        super().__init__()
        self.byte_emb = nn.Embedding(257, d_byte)
        self.conv = nn.Conv1d(d_byte * 8, d_payload, kernel_size=3, padding=1)

    def forward(self, payload: torch.Tensor) -> torch.Tensor:
        x = self.byte_emb(payload).flatten(-2).transpose(1, 2)
        return self.conv(x).transpose(1, 2)


class TimeEncoder(nn.Module):
    def __init__(self, d_time=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, d_time))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(t)


class TransitionContextEncoder(nn.Module):
    def __init__(self, id_emb: nn.Embedding, d_state=64):
        super().__init__()
        self.id_emb = id_emb
        self.proj = nn.Linear(id_emb.embedding_dim, d_state)

    def forward(self, neighbors: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        emb = self.id_emb(neighbors)
        ctx = (emb * weights.unsqueeze(-1)).sum(-2)
        return self.proj(ctx)
