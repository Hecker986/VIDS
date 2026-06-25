"""Build transition graph for hcrl_sa directly."""
import sys, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, "/root/autodl-tmp/scs-can")
from src.utils.schema import SPLIT_TRAIN, TOP_K_TRANSITION, VOCAB_SIZE

frames = pd.read_parquet("/root/autodl-tmp/scs-can/data/processed/hcrl_sa/frames.parquet")
idx = np.load("/root/autodl-tmp/scs-can/data/processed/hcrl_sa/windows_index.npy")
train = idx[idx[:, 4] == SPLIT_TRAIN]
print(f"Train windows: {len(train)}")
# vectorized: collect all train frame indices at once
all_pos = np.concatenate([np.arange(s, e) for s, e, *_ in train])
ids = frames["can_id"].values[all_pos].astype(np.int64)
print(f"Train frames: {len(ids)}")
counts = np.zeros((VOCAB_SIZE, VOCAB_SIZE), np.float64)
np.add.at(counts, (ids[:-1], ids[1:]), 1)
neigh = np.argsort(-counts, axis=1)[:, :TOP_K_TRANSITION]
w = np.take_along_axis(counts, neigh, 1)
w = w / np.maximum(w.sum(1, keepdims=True), 1)
out = Path("/root/autodl-tmp/scs-can/data/processed/hcrl_sa")
np.save(out / "transition_neighbors.npy", neigh.astype(np.int32))
np.save(out / "transition_weights.npy", w.astype(np.float32))
print("saved transition graph")
