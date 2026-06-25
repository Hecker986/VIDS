"""Transformer baseline without transition context."""
from __future__ import annotations

import torch
from torch import nn


class CANTransformer(nn.Module):
    def __init__(self, d_model=256, num_layers=4, num_classes=2):
        super().__init__()
        self.id_emb = nn.Embedding(4098, 64)
        self.byte_emb = nn.Embedding(256, 16)
        self.in_proj = nn.Linear(64 + 128, d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=512,
                                           dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Sequential(nn.Linear(d_model, 128), nn.ReLU(), nn.Linear(128, num_classes))
        self.pay_proj = nn.Sequential(nn.Linear(128, 128), nn.ReLU())

    def _payload(self, payload: torch.Tensor) -> torch.Tensor:
        return self.pay_proj(self.byte_emb(payload).flatten(-2))

    def forward(self, batch: dict) -> torch.Tensor:
        x = torch.cat([self.id_emb(batch["can_id"]), self._payload(batch["payload"])], dim=-1)
        h = self.encoder(self.in_proj(x))
        return self.head(h.mean(1))
