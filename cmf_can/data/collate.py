from __future__ import annotations

import torch


def collate_batch(batch: list[dict]) -> dict:
    out = {}
    for key in batch[0].keys():
        values = [item[key] for item in batch]
        if torch.is_tensor(values[0]):
            out[key] = torch.stack(values)
        else:
            out[key] = values
    return out
