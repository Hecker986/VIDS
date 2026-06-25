"""Train-split CAN ID transition top-k neighbors."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.schema import SPLIT_TRAIN, TOP_K_TRANSITION, VOCAB_SIZE


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--windows", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--top-k", type=int, default=TOP_K_TRANSITION)
    args = ap.parse_args()
    frames = pd.read_parquet(args.frames)
    idx = np.load(args.windows)
    train = idx[idx[:, 4] == SPLIT_TRAIN]
    ids = []
    for s, e, *_ in train:
        ids.extend(frames["can_id"].iloc[s:e].tolist())
    ids = np.asarray(ids, dtype=np.int64)
    counts = np.zeros((VOCAB_SIZE, VOCAB_SIZE), np.float64)
    np.add.at(counts, (ids[:-1], ids[1:]), 1)
    neigh = np.argsort(-counts, axis=1)[:, :args.top_k]
    w = np.take_along_axis(counts, neigh, 1)
    w = w / np.maximum(w.sum(1, keepdims=True), 1)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "transition_neighbors.npy", neigh.astype(np.int32))
    np.save(out / "transition_weights.npy", w.astype(np.float32))
    print(f"saved transition graph top_k={args.top_k}", flush=True)


if __name__ == "__main__":
    main()
