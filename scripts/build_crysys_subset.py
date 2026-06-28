"""Build a documented CrySyS subset from a processed full CrySyS parquet."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.schema import (  # noqa: E402
    ATTACK_PRIORITY, ATTACK_TO_ID, SPLIT_TEST, SPLIT_TRAIN, SPLIT_VAL,
    WINDOW_SIZE, contiguous_runs,
)


DEFAULT_GROUPS = ["S-1-1", "S-1-2", "S-1-3", "S-1-4", "S-1-5", "S-2-1"]


def read_filtered_parquet(path: Path, groups: list[str]) -> pd.DataFrame:
    selected = pa.array(groups)
    chunks = []
    pf = pq.ParquetFile(path)
    for i in range(pf.metadata.num_row_groups):
        table = pf.read_row_group(i)
        mask = pc.is_in(table["split_group"], value_set=selected)
        if pc.any(mask).as_py():
            chunks.append(table.filter(mask))
    if not chunks:
        raise ValueError(f"no rows matched selected groups: {groups}")
    return pa.concat_tables(chunks).to_pandas().reset_index(drop=True)


def build_windows(df: pd.DataFrame, split_col: np.ndarray) -> np.ndarray:
    rows = []
    for split_id in [SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST]:
        pos = np.where(split_col == split_id)[0]
        for run in contiguous_runs(pos):
            if len(run) < WINDOW_SIZE:
                continue
            for i in range(0, len(run) - WINDOW_SIZE + 1, WINDOW_SIZE):
                start = int(run[i])
                end = int(run[i + WINDOW_SIZE - 1]) + 1
                chunk = df.iloc[start:end]
                labels = chunk["label"].to_numpy()
                attacks = chunk["attack_type"].tolist()
                y = int(labels.max())
                if y:
                    attack = max(
                        (a for a in attacks if a != "normal"),
                        key=lambda a: ATTACK_PRIORITY.get(a, -1),
                        default="normal",
                    )
                else:
                    attack = "normal"
                rows.append([start, end, y, ATTACK_TO_ID.get(attack, ATTACK_TO_ID["unknown"]), split_id])
    return np.asarray(rows, dtype=np.int64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--source", default="data/processed/crysys/frames.parquet")
    ap.add_argument("--dataset", default="crysys_subset")
    ap.add_argument("--groups", nargs="+", default=DEFAULT_GROUPS)
    args = ap.parse_args()

    root = Path(args.root)
    source = root / args.source
    out = root / "data/processed" / args.dataset
    out.mkdir(parents=True, exist_ok=True)

    print(f"building {args.dataset} from {source}", flush=True)
    df = read_filtered_parquet(source, args.groups)
    split_col = np.full(len(df), -1, dtype=np.int64)
    split_ranges = []
    for group in args.groups:
        idx = np.where((df["split_group"].astype(str) == group).to_numpy())[0]
        n_train = int(len(idx) * 0.7)
        n_val = int(len(idx) * 0.1)
        assignments = [
            (SPLIT_TRAIN, idx[:n_train], 0, n_train),
            (SPLIT_VAL, idx[n_train:n_train + n_val], n_train, n_train + n_val),
            (SPLIT_TEST, idx[n_train + n_val:], n_train + n_val, len(idx)),
        ]
        for split_id, sub_idx, rel_start, rel_end in assignments:
            split_col[sub_idx] = split_id
            if len(sub_idx):
                split_ranges.append({
                    "split_group": group,
                    "split": {SPLIT_TRAIN: "train", SPLIT_VAL: "val", SPLIT_TEST: "test"}[split_id],
                    "start": int(rel_start),
                    "end": int(rel_end),
                })

    df.to_parquet(out / "frames.parquet", index=False)
    windows = build_windows(df, split_col)
    np.save(out / "windows_index.npy", windows)
    (out / "subset_manifest.json").write_text(json.dumps({
        "source": str(source),
        "selected_split_groups": args.groups,
        "frames": int(len(df)),
        "attack_frames": int(df["label"].sum()),
        "attack_rate": float(df["label"].mean()),
    }, indent=2))
    (out / "splits.json").write_text(json.dumps({"groups": {g: "mixed" for g in args.groups},
                                                 "ranges": split_ranges}, indent=2))

    commands = [
        [sys.executable, "-m", "src.preprocessing.build_transition_graph",
         "--frames", str(out / "frames.parquet"),
         "--windows", str(out / "windows_index.npy"),
         "--out-dir", str(out)],
        [sys.executable, "-m", "src.preprocessing.compute_train_stats",
         "--frames", str(out / "frames.parquet"),
         "--windows", str(out / "windows_index.npy"),
         "--out", str(out / "train_stats.json")],
        [sys.executable, "-m", "src.preprocessing.audit_processed_dataset",
         "--root", str(root),
         "--dataset", args.dataset],
    ]
    for cmd in commands:
        subprocess.run(cmd, check=True, cwd=str(root))
    print(f"built {args.dataset}: frames={len(df)} windows={len(windows)}", flush=True)


if __name__ == "__main__":
    main()
