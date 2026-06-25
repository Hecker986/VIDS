"""Build non-overlapping window index from frames + splits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.schema import (
    ATTACK_PRIORITY, ATTACK_TO_ID, SPLIT_TEST, SPLIT_TRAIN, SPLIT_VAL, STRIDE,
    WINDOW_SIZE, contiguous_runs,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--window", type=int, default=WINDOW_SIZE)
    ap.add_argument("--stride", type=int, default=STRIDE)
    args = ap.parse_args()
    df = pd.read_parquet(args.frames).reset_index(drop=True)
    spec = json.loads(Path(args.splits).read_text())
    split_map = {"train": SPLIT_TRAIN, "val": SPLIT_VAL, "test": SPLIT_TEST}
    rows = []
    for rec in spec["ranges"]:
        g, split = rec["split_group"], rec["split"]
        sid = split_map[split]
        pos = np.where(df["split_group"].astype(str).values == g)[0]
        pos = pos[rec["start"]:rec["end"]]
        for run in contiguous_runs(pos):
            if len(run) < args.window:
                continue
            for i in range(0, len(run) - args.window + 1, args.stride):
                gstart = int(run[i])
                gend = int(run[i + args.window - 1]) + 1
                chunk = df.iloc[gstart:gend]
                labels = chunk["label"].to_numpy()
                attacks = chunk["attack_type"].tolist()
                y = int(labels.max())
                if y:
                    at = max((a for a in attacks if a != "normal"),
                             key=lambda a: ATTACK_PRIORITY.get(a, -1), default="normal")
                else:
                    at = "normal"
                rows.append([gstart, gend, y, ATTACK_TO_ID[at], sid])
    idx = np.asarray(rows, dtype=np.int64)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, idx)
    print(f"windows {len(idx)}", flush=True)
    for sid, name in [(0, "train"), (1, "val"), (2, "test")]:
        m = idx[:, 4] == sid
        print(f"{name} {m.sum()} attack {idx[m, 2].sum()}", flush=True)


if __name__ == "__main__":
    main()
