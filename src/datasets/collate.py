"""Batch collation for variable-length windows (fixed window_size=100)."""
from __future__ import annotations

import torch


def collate_batch(batch: list[dict]) -> dict:
    keys = batch[0].keys()
    out = {}
    for k in keys:
        if k == "label" or k == "ipc_match":
            out[k] = torch.stack([b[k] for b in batch])
        else:
            out[k] = torch.stack([b[k] for b in batch])
    return out
