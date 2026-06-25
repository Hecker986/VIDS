"""1D-CNN baseline on flattened CAN window."""
from __future__ import annotations

import torch
from torch import nn


class CANCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(9, 64, 3, padding=1), nn.ReLU(),
            nn.Conv1d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(128, num_classes)

    def forward(self, batch: dict) -> torch.Tensor:
        cid = batch["can_id"].float().unsqueeze(-1) / 4096.0
        pay = batch["payload"].float() / 255.0
        x = torch.cat([cid, pay], dim=-1).transpose(1, 2)
        h = self.net(x).squeeze(-1)
        return self.head(h)
