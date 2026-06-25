"""Create CrySyS subset using pyarrow filters (no full load)."""
import sys, json, time
import numpy as np
import pyarrow.parquet as pq
import pyarrow.compute as pc
import pandas as pd
from pathlib import Path

sys.path.insert(0, "/root/autodl-tmp/scs-can")
from src.utils.schema import SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST, VOCAB_SIZE, TOP_K_TRANSITION

SRC = Path("/root/autodl-tmp/scs-can/data/processed/crysys/frames.parquet")
DST = Path("/root/autodl-tmp/scs-can/data/processed/crysys_subset")
DST.mkdir(parents=True, exist_ok=True)

selected = ["S-1-1", "S-1-2", "S-1-3", "S-1-4", "S-1-5", "S-2-1"]

t0 = time.time()
print("Reading with pyarrow filter...", flush=True)
table = pq.read_table(SRC, filters=[("split_group", "in", selected)])
print(f"  Read {table.num_rows} rows in {time.time()-t0:.1f}s", flush=True)

sub = table.to_pandas()
del table
print(f"Subset: {len(sub)} frames, {sub.label.sum()} attacks ({sub.label.mean()*100:.2f}%)", flush=True)

# Time-block splits per group (70/10/20)
split_col = np.full(len(sub), -1, dtype=np.int64)
for g in selected:
    mask = (sub.split_group == g).values
    idxs = np.where(mask)[0]
    n = len(idxs)
    n_train = int(n * 0.7)
    n_val = int(n * 0.1)
    split_col[idxs[:n_train]] = SPLIT_TRAIN
    split_col[idxs[n_train:n_train+n_val]] = SPLIT_VAL
    split_col[idxs[n_train+n_val:]] = SPLIT_TEST

sub["split"] = split_col
sub.to_parquet(DST / "frames.parquet", index=False)
print("Saved frames.parquet", flush=True)

# Build window index
WINDOW, STRIDE = 100, 100
windows = []
for sp in [SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST]:
    sp_idxs = np.where(split_col == sp)[0]
    if len(sp_idxs) == 0:
        continue
    breaks = np.where(np.diff(sp_idxs) != 1)[0] + 1
    runs = np.split(sp_idxs, breaks)
    for run in runs:
        for i in range(0, len(run) - WINDOW + 1, STRIDE):
            s = int(run[i])
            e = int(run[i + WINDOW - 1]) + 1
            lbl = int(sub.iloc[s:e]["label"].any())
            windows.append([s, e, lbl, 0, sp])

windows = np.array(windows, dtype=np.int64)
np.save(DST / "windows_index.npy", windows)
for sp, nm in [(SPLIT_TRAIN, "train"), (SPLIT_VAL, "val"), (SPLIT_TEST, "test")]:
    sw = windows[windows[:, 4] == sp]
    print(f"  {nm}: {len(sw)} windows, {sw[:, 2].sum()} attack", flush=True)

# Transition graph
train_w = windows[windows[:, 4] == SPLIT_TRAIN]
all_pos = np.concatenate([np.arange(s, e) for s, e in train_w[:, :2]])
ids = sub["can_id"].values[all_pos].astype(np.int64)
counts = np.zeros((VOCAB_SIZE, VOCAB_SIZE), np.float64)
np.add.at(counts, (ids[:-1], ids[1:]), 1)
neigh = np.argsort(-counts, axis=1)[:, :TOP_K_TRANSITION]
w = np.take_along_axis(counts, neigh, 1)
w = w / np.maximum(w.sum(1, keepdims=True), 1)
np.save(DST / "transition_neighbors.npy", neigh.astype(np.int32))
np.save(DST / "transition_weights.npy", w.astype(np.float32))
print("Saved transition graph", flush=True)

# Train stats
train_stats = {
    "vocab_size": VOCAB_SIZE,
    "n_unique_ids": int(len(set(sub["can_id"].values[split_col == SPLIT_TRAIN].tolist()))),
    "n_train_frames": int((split_col == SPLIT_TRAIN).sum()),
}
with open(DST / "train_stats.json", "w") as f:
    json.dump(train_stats, f)

# Splits json
splits = {str(i): int(v) for i, v in enumerate(split_col)}
with open(DST / "splits.json", "w") as f:
    json.dump(splits, f)

print(f"\n=== DONE in {time.time()-t0:.1f}s: CrySyS subset at {DST} ===", flush=True)
