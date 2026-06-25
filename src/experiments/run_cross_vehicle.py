"""Cross-vehicle HCRL-SA experiments.

Train on one vehicle, validate on a portion of the same vehicle,
test on a different vehicle.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.training.trainer import finetune, pretrain
from src.utils.schema import (
    SPLIT_TEST, SPLIT_TRAIN, SPLIT_VAL,
    ATTACK_TO_ID, ATTACK_PRIORITY, contiguous_runs,
)


def build_vehicle_splits(frames_path: Path, train_v: str, test_v: str) -> None:
    """Build window_index for cross-vehicle: train_v -> train+val, test_v -> test."""
    df = pd.read_parquet(frames_path)
    proc = frames_path.parent
    window, stride = 100, 100
    rows = []

    for veh in df["vehicle"].unique():
        pos = np.where(df["vehicle"].values == veh)[0]
        if veh == train_v:
            # 80% train, 20% val within this vehicle
            n = len(pos)
            n_train = int(n * 0.8)
            train_pos = pos[:n_train]
            val_pos = pos[n_train + 100:]  # buffer
            for split_id, sub_pos in [(SPLIT_TRAIN, train_pos), (SPLIT_VAL, val_pos)]:
                if len(sub_pos) == 0:
                    continue
                for run in contiguous_runs(sub_pos):
                    if len(run) < window:
                        continue
                    for i in range(0, len(run) - window + 1, stride):
                        gs = int(run[i])
                        ge = int(run[i + window - 1]) + 1
                        chunk = df.iloc[gs:ge]
                        y = int(chunk["label"].max())
                        at = "normal"
                        if y:
                            attacks = chunk["attack_type"].tolist()
                            at = max((a for a in attacks if a != "normal"),
                                     key=lambda a: ATTACK_PRIORITY.get(a, -1), default="normal")
                        rows.append([gs, ge, y, ATTACK_TO_ID[at], split_id])
        elif veh == test_v:
            for run in contiguous_runs(pos):
                if len(run) < window:
                    continue
                for i in range(0, len(run) - window + 1, stride):
                    gs = int(run[i])
                    ge = int(run[i + window - 1]) + 1
                    chunk = df.iloc[gs:ge]
                    y = int(chunk["label"].max())
                    at = "normal"
                    if y:
                        attacks = chunk["attack_type"].tolist()
                        at = max((a for a in attacks if a != "normal"),
                                 key=lambda a: ATTACK_PRIORITY.get(a, -1), default="normal")
                    rows.append([gs, ge, y, ATTACK_TO_ID[at], SPLIT_TEST])

    idx = np.asarray(rows, dtype=np.int64)
    np.save(proc / "windows_index.npy", idx)
    for sid, name in [(0, "train"), (1, "val"), (2, "test")]:
        m = idx[:, 4] == sid
        atk = idx[m, 2].sum()
        print(f"  {name}: {m.sum()} windows, attack={atk}", flush=True)

    # rebuild transition graph from train split only
    import subprocess
    subprocess.run([sys.executable, "-m", "src.preprocessing.build_transition_graph",
                    
                    "--frames", str(frames_path),
                    "--windows", str(proc / "windows_index.npy"),
                    "--out-dir", str(proc)], check=True, cwd=str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--train-vehicle", default="kia")
    ap.add_argument("--test-vehicle", default="sonata")
    args = ap.parse_args()
    root = Path(args.root)
    frames = root / "data/processed/hcrl_sa/frames.parquet"

    print(f"Cross-vehicle: train={args.train_vehicle}, test={args.test_vehicle}", flush=True)
    build_vehicle_splits(frames, args.train_vehicle, args.test_vehicle)

    out = root / "results/tables/cross_vehicle_results.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out.exists()

    # pretrain on train vehicle first
    print("Pretraining on train vehicle...", flush=True)
    pt_path = pretrain(root, "hcrl_sa", epochs=10, seed=42)

    for model, variant in [("transformer", "full"), ("scs_can", "wo_ssl"), ("scs_can", "full")]:
        pt = pt_path if (variant == "full" and model == "scs_can") else None
        ipc_w = 0.1 if (variant == "full" and model == "scs_can") else 0.0
        r = finetune(root, "hcrl_sa", model,
                     variant=variant if model == "scs_can" else "full",
                     epochs=15, pretrained=pt, ipc_weight=ipc_w)
        r["train_vehicle"] = args.train_vehicle
        r["test_vehicle"] = args.test_vehicle
        with out.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(r.keys()))
            if write_header:
                w.writeheader()
                write_header = False
            w.writerow(r)
        print(f"  {model}/{variant}: F1={r.get('macro_f1', 'N/A')}", flush=True)


if __name__ == "__main__":
    main()
