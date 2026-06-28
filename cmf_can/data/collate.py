from __future__ import annotations

import torch


def collate_batch(batch: list[dict]) -> dict:
    return {k: torch.stack([item[k] for item in batch]) for k in batch[0].keys()}

