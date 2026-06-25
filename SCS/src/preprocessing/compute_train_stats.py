"""Train-split normalization stats for time features."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.schema import SPLIT_TRAIN


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--windows", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    frames = pd.read_parquet(args.frames)
    idx = np.load(args.windows)
    train = idx[idx[:, 4] == SPLIT_TRAIN]
    vals = []
    for s, e, *_ in train:
        sub = frames.iloc[s:e]
        vals.append(sub[["delta_t_global", "delta_t_same_id"]].to_numpy())
    x = np.concatenate(vals, axis=0)
    stats = {"mean": x.mean(0).tolist(), "std": np.maximum(x.std(0), 1e-6).tolist()}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(stats, indent=2))
    print(stats, flush=True)


if __name__ == "__main__":
    main()
