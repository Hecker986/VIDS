"""LSTM baseline."""
from __future__ import annotations

import torch
from torch import nn


class CANLSTM(nn.Module):
    def __init__(self, num_classes=2, hidden=128):
        super().__init__()
        self.in_proj = nn.Linear(9, hidden)
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True, num_layers=2, dropout=0.1)
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, batch: dict) -> torch.Tensor:
        cid = batch["can_id"].float().unsqueeze(-1) / 4096.0
        pay = batch["payload"].float() / 255.0
        x = torch.cat([cid, pay], dim=-1)
        h, _ = self.lstm(self.in_proj(x))
        return self.head(h.mean(1))
