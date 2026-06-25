"""Dynamic window dataset — reads frames[start:end] on the fly."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class DynamicWindowDataset(Dataset):
    def __init__(self, frames: pd.DataFrame, windows: np.ndarray, split_id: int,
                 neighbors: np.ndarray, weights: np.ndarray, time_stats: dict | None = None):
        self.frames = frames
        self.windows = windows[windows[:, 4] == split_id]
        self.neighbors = neighbors
        self.weights = weights
        self.time_stats = time_stats
        cid = frames["can_id"].to_numpy(copy=True)
        pay = np.stack([frames[f"data{i}"].to_numpy(copy=True) for i in range(8)], axis=1)
        dlc = frames["dlc"].to_numpy(copy=True)
        dtg = frames["delta_t_global"].to_numpy(dtype=np.float32, copy=True)
        dts = frames["delta_t_same_id"].to_numpy(dtype=np.float32, copy=True)
        self.cid, self.pay, self.dlc = cid, pay, dlc
        self.time = np.stack([dtg, dts], axis=1)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict:
        s, e, y, _, _ = self.windows[idx]
        cid = torch.as_tensor(self.cid[s:e], dtype=torch.long)
        pay = torch.as_tensor(self.pay[s:e], dtype=torch.long)
        dlc = torch.as_tensor(self.dlc[s:e], dtype=torch.long)
        time = torch.as_tensor(self.time[s:e], dtype=torch.float32)
        if self.time_stats:
            mean = torch.tensor(self.time_stats["mean"], dtype=torch.float32)
            std = torch.tensor(self.time_stats["std"], dtype=torch.float32)
            time = (time - mean) / std
        n = torch.as_tensor(self.neighbors[cid.numpy()], dtype=torch.long)
        w = torch.as_tensor(self.weights[cid.numpy()], dtype=torch.float32)
        return {"can_id": cid, "payload": pay, "dlc": dlc, "time_features": time,
                "neighbors": n, "weights": w, "label": torch.tensor(y, dtype=torch.long)}


def load_dataset_bundle(root: Path, dataset: str, split_id: int) -> DynamicWindowDataset:
    proc = root / "data/processed" / dataset
    frames = pd.read_parquet(proc / "frames.parquet")
    windows = np.load(proc / "windows_index.npy")
    neigh = np.load(proc / "transition_neighbors.npy")
    w = np.load(proc / "transition_weights.npy")
    stats_path = proc / "train_stats.json"
    stats = None
    if stats_path.exists():
        import json
        stats = json.loads(stats_path.read_text())
    return DynamicWindowDataset(frames, windows, split_id, neigh, w, stats)
