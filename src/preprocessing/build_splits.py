"""Capture-level or time-block train/val/test splits with buffer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.schema import WINDOW_SIZE


def assign_splits_capture(groups: list[str], train=0.7, val=0.1) -> dict[str, str]:
    uniq = sorted(set(groups))
    n = len(uniq)
    n_train = max(1, int(n * train))
    n_val = max(1, int(n * val)) if n >= 3 else 0
    mapping = {}
    for i, g in enumerate(uniq):
        if i < n_train:
            mapping[g] = "train"
        elif i < n_train + n_val:
            mapping[g] = "val"
        else:
            mapping[g] = "test"
    return mapping


def build_time_block_splits(df: pd.DataFrame, buffer: int,
                            train_r=0.7, val_r=0.1) -> dict:
    """Split ALL frames by time: first 70% train, next 10% val, last 20% test.

    Each capture is split independently to avoid mixing captures at boundaries.
    """
    groups_used = sorted(df["split_group"].astype(str).unique())
    mapping = {}
    ranges = []
    for g in groups_used:
        pos = np.where(df["split_group"].astype(str).values == g)[0]
        n = len(pos)
        if n < buffer * 3:
            mapping[g] = "train"
            ranges.append({"split_group": g, "split": "train", "start": 0, "end": n})
            continue
        n_train = int(n * train_r)
        n_val = int(n * val_r)
        mapping[g] = "mixed"
        # train: [0, n_train - buffer)
        if n_train - buffer > 0:
            ranges.append({"split_group": g, "split": "train",
                           "start": 0, "end": n_train - buffer})
        # val: [n_train + buffer, n_train + n_val - buffer)
        v_start = n_train + buffer
        v_end = n_train + n_val - buffer
        if v_end > v_start:
            ranges.append({"split_group": g, "split": "val",
                           "start": v_start, "end": v_end})
        # test: [n_train + n_val + buffer, n)
        t_start = n_train + n_val + buffer
        if t_start < n:
            ranges.append({"split_group": g, "split": "test",
                           "start": t_start, "end": n})
    return {"groups": mapping, "ranges": ranges}


def build_capture_splits(df: pd.DataFrame, buffer: int) -> dict:
    groups = df["split_group"].astype(str).unique().tolist()
    mapping = assign_splits_capture(groups)
    ranges = []
    for g, split in mapping.items():
        pos = np.where(df["split_group"].astype(str).values == g)[0]
        n = len(pos)
        if n <= buffer * 2:
            ranges.append({"split_group": g, "split": split, "start": 0, "end": n})
            continue
        if split == "train":
            ranges.append({"split_group": g, "split": split, "start": 0, "end": n - buffer})
        elif split == "val":
            ranges.append({"split_group": g, "split": split, "start": buffer, "end": n - buffer})
        else:
            ranges.append({"split_group": g, "split": split, "start": buffer, "end": n})
    return {"groups": mapping, "ranges": ranges}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--buffer", type=int, default=WINDOW_SIZE)
    ap.add_argument("--strategy", default="capture", choices=["capture", "time_block"])
    args = ap.parse_args()
    df = pd.read_parquet(args.frames).reset_index(drop=True)

    if args.strategy == "time_block":
        result = build_time_block_splits(df, args.buffer)
    else:
        result = build_capture_splits(df, args.buffer)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))

    # summary
    from collections import Counter
    split_counts = Counter()
    for r in result["ranges"]:
        split_counts[r["split"]] += r["end"] - r["start"]
    for s in ["train", "val", "test"]:
        print(f"  {s}: {split_counts.get(s, 0)} frames", flush=True)


if __name__ == "__main__":
    main()
