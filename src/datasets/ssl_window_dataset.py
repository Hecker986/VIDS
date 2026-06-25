"""SSL dataset with MFM masking and IPC pairs."""
from __future__ import annotations

import random

import torch
from torch.utils.data import Dataset

from src.datasets.dynamic_window_dataset import DynamicWindowDataset


class SSLWindowDataset(Dataset):
    def __init__(self, base: DynamicWindowDataset, mask_id_prob=0.15, mask_payload_prob=0.15):
        self.base = base
        self.mask_id_prob = mask_id_prob
        self.mask_payload_prob = mask_payload_prob

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict:
        item = self.base[idx]
        out = {k: v.clone() if torch.is_tensor(v) else v for k, v in item.items()}
        cid, pay = out["can_id"], out["payload"]
        L = cid.size(0)
        id_mask = torch.rand(L) < self.mask_id_prob
        pay_mask = torch.rand(L, 8) < self.mask_payload_prob
        out["id_mask"] = id_mask
        out["pay_mask"] = pay_mask
        out["target_id"] = cid.clone()
        out["target_pay"] = pay.clone()
        cid[id_mask] = 4097
        pay[pay_mask] = 256
        # IPC: half match, half mismatch
        match = torch.tensor(random.random() < 0.5)
        if not match and L > 1:
            j = random.randrange(L)
            pay[:, :] = self.base[random.randrange(len(self.base))]["payload"][j].unsqueeze(0).expand(L, -1)
        out["ipc_match"] = match.long()
        return out
