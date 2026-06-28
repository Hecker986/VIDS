"""Window dataset for CMF-CAN features."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset

ID_SIZE = 4098
SPLIT_TRAIN = 0
SPLIT_VAL = 1
SPLIT_TEST = 2


class CMFWindowDataset(Dataset):
    def __init__(self, root: Path, dataset: str, split_id: int):
        proc = root / "data/processed" / dataset
        feature_dir = proc / "cmf_features"
        if not feature_dir.exists():
            raise FileNotFoundError(f"missing CMF features: {feature_dir}")
        frames = pd.read_parquet(proc / "frames.parquet")
        windows = np.load(proc / "windows_index.npy")
        self.row_indices = np.where(windows[:, 4] == split_id)[0].astype(np.int64)
        self.windows = windows[self.row_indices].astype(np.int64)
        self.can_id = frames["can_id"].to_numpy(np.int64, copy=True).clip(0, ID_SIZE - 1)
        self.attack_type = frames["attack_type"].astype(str).to_numpy(copy=True) if "attack_type" in frames else None
        self.vehicle = frames["vehicle"].astype(str).to_numpy(copy=True) if "vehicle" in frames else None
        self.payload = np.stack([frames[f"data{i}"].to_numpy(np.uint8, copy=True) for i in range(8)], axis=1)
        self.frame_numeric = np.load(feature_dir / "frame_numeric.npy", mmap_mode="r")
        self.window_stats = np.load(feature_dir / "window_stats.npy", mmap_mode="r")
        self.id_context = np.load(feature_dir / "id_context.npy", mmap_mode="r")

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict:
        start, end, label, attack_code, _ = self.windows[idx]
        ids = self.can_id[start:end]
        ctx = np.asarray(self.id_context[ids]).copy()
        model_ids = ids.copy()
        unseen = np.isclose(np.abs(ctx).sum(axis=1), 0.0)
        model_ids[unseen] = ID_SIZE - 1
        attack_type = "NA"
        if self.attack_type is not None:
            attacks = self.attack_type[start:end]
            malicious = attacks[attacks != "normal"]
            attack_type = str(malicious[0] if len(malicious) else attacks[0])
        vehicle = "NA"
        if self.vehicle is not None:
            vehicles, counts = np.unique(self.vehicle[start:end], return_counts=True)
            vehicle = str(vehicles[int(np.argmax(counts))])
        return {
            "can_id": torch.as_tensor(model_ids, dtype=torch.long),
            "payload": torch.as_tensor(self.payload[start:end], dtype=torch.long),
            "frame_numeric": torch.as_tensor(np.asarray(self.frame_numeric[start:end]).copy(), dtype=torch.float32),
            "window_stats": torch.as_tensor(np.asarray(self.window_stats[self.row_indices[idx]]).copy(), dtype=torch.float32),
            "id_context": torch.as_tensor(ctx, dtype=torch.float32),
            "label": torch.tensor(int(label), dtype=torch.long),
            "sample_id": f"{self.row_indices[idx]}",
            "attack_type": attack_type,
            "vehicle": vehicle,
            "window_start": int(start),
            "window_end": int(end),
            "split": {SPLIT_TRAIN: "train", SPLIT_VAL: "val", SPLIT_TEST: "test"}.get(int(self.windows[idx, 4]), "NA"),
        }


def stratified_subset(dataset: CMFWindowDataset, ratio: float, seed: int) -> Dataset:
    if ratio >= 1.0:
        return dataset
    labels = dataset.windows[:, 2]
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for cls in (0, 1):
        idx = np.where(labels == cls)[0]
        if len(idx) == 0:
            continue
        n = max(1, int(round(len(idx) * ratio)))
        selected.extend(rng.choice(idx, size=min(n, len(idx)), replace=False).tolist())
    rng.shuffle(selected)
    return Subset(dataset, selected)
