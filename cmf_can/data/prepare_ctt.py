"""Prepare CT&T set_01 official generalization splits for CMF-CAN."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SPLIT_TRAIN = 0
SPLIT_VAL = 1
SPLIT_TEST = 2

TEST_FOLDERS = {
    "ctt_test01": "set_01/test_01_known_vehicle_known_attack",
    "ctt_test02": "set_01/test_02_unknown_vehicle_known_attack",
    "ctt_test03": "set_01/test_03_known_vehicle_unknown_attack",
    "ctt_test04": "set_01/test_04_unknown_vehicle_unknown_attack",
}


def _parse_payload(series: pd.Series) -> np.ndarray:
    values = series.fillna("").astype(str).str.replace(" ", "", regex=False).str.upper()
    out = np.zeros((len(values), 8), dtype=np.uint8)
    for i, text in enumerate(values):
        text = text[:16].ljust(16, "0")
        for j in range(8):
            try:
                out[i, j] = int(text[j * 2:j * 2 + 2], 16)
            except ValueError:
                out[i, j] = 0
    return out


def _parse_can_id(series: pd.Series) -> np.ndarray:
    def parse_one(x) -> int:
        text = str(x).strip()
        try:
            return int(text, 16)
        except ValueError:
            return int(float(text))
    return series.map(parse_one).to_numpy(np.int64)


def _read_file(path: Path, split_id: int, capture_id: str, val_fraction: float) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "arbitration_id" not in df.columns or "data_field" not in df.columns or "attack" not in df.columns:
        raise ValueError(f"unexpected CT&T schema in {path}: {df.columns.tolist()}")
    ts = df["timestamp"].astype(float).to_numpy(np.float64)
    order = np.argsort(ts, kind="stable")
    df = df.iloc[order].reset_index(drop=True)
    ts = ts[order]
    can_id = _parse_can_id(df["arbitration_id"])
    payload = _parse_payload(df["data_field"])
    label = df["attack"].astype(int).clip(0, 1).to_numpy(np.int64)
    n = len(df)
    split = np.full(n, split_id, dtype=np.int64)
    group_suffix = "test"
    if split_id != SPLIT_TEST:
        if path.stem.endswith("-2"):
            split[:] = SPLIT_VAL
            group_suffix = "val"
        else:
            split[:] = SPLIT_TRAIN
            group_suffix = "train"
    delta_global = np.zeros(n, dtype=np.float32)
    if n > 1:
        delta_global[1:] = np.maximum(np.diff(ts), 0).astype(np.float32)
    delta_same = np.zeros(n, dtype=np.float32)
    last_ts: dict[int, float] = {}
    for i, (cid, t) in enumerate(zip(can_id, ts)):
        if int(cid) in last_ts:
            delta_same[i] = max(float(t - last_ts[int(cid)]), 0.0)
        last_ts[int(cid)] = float(t)
    out = pd.DataFrame(
        {
            "timestamp": ts,
            "can_id": can_id,
            "dlc": np.full(n, 8, dtype=np.int64),
            "delta_t_global": delta_global,
            "delta_t_same_id": delta_same,
            "label": label,
            "attack_type": np.where(label == 1, path.stem, "normal"),
            "dataset": "ctt",
            "vehicle": "unknown",
            "capture_id": capture_id,
            "split_group": [f"{capture_id}:{group_suffix}:{int(s)}" for s in split],
            "split_id": split,
        }
    )
    for i in range(8):
        out[f"data{i}"] = payload[:, i]
    cols = [
        "timestamp", "can_id", "dlc", "data0", "data1", "data2", "data3", "data4",
        "data5", "data6", "data7", "delta_t_global", "delta_t_same_id", "label",
        "attack_type", "dataset", "vehicle", "capture_id", "split_group", "split_id",
    ]
    return out[cols]


def _build_windows(frames: pd.DataFrame, window_size: int, stride: int) -> np.ndarray:
    windows: list[list[int]] = []
    attack_type_codes, _ = pd.factorize(frames["attack_type"], sort=True)
    for _, idx in frames.groupby("split_group", sort=False).indices.items():
        arr = np.asarray(idx, dtype=np.int64)
        split_id = int(frames.iloc[arr[0]]["split_id"])
        labels = frames["label"].to_numpy(np.int64)[arr]
        attack_codes = attack_type_codes[arr]
        for off in range(0, len(arr) - window_size + 1, stride):
            win_idx = arr[off:off + window_size]
            win_labels = labels[off:off + window_size]
            y = int(win_labels.max())
            atk = int(attack_codes[off:off + window_size][win_labels.astype(bool)][0]) if y else 0
            windows.append([int(win_idx[0]), int(win_idx[-1] + 1), y, atk, split_id])
    return np.asarray(windows, dtype=np.int64)


def prepare_one(
    raw_root: Path,
    out_root: Path,
    name: str,
    test_folder: str,
    window_size: int,
    stride: int,
    val_fraction: float,
) -> None:
    train_files = sorted((raw_root / "set_01/train_01").glob("*.csv"))
    test_files = sorted((raw_root / test_folder).glob("*.csv"))
    if not train_files:
        raise FileNotFoundError(raw_root / "set_01/train_01")
    if not test_files:
        raise FileNotFoundError(raw_root / test_folder)
    parts = []
    for path in train_files:
        parts.append(_read_file(path, SPLIT_TRAIN, f"train_01/{path.name}", val_fraction))
        print(f"[ctt] read train {path}", flush=True)
    for path in test_files:
        parts.append(_read_file(path, SPLIT_TEST, f"{test_folder}/{path.name}", val_fraction))
        print(f"[ctt] read test {path}", flush=True)
    frames = pd.concat(parts, ignore_index=True)
    split_id = frames.pop("split_id").to_numpy(np.int64)
    frames.insert(len(frames.columns), "split_id_tmp", split_id)
    windows = _build_windows(frames.rename(columns={"split_id_tmp": "split_id"}), window_size, stride)
    frames = frames.drop(columns=["split_id_tmp"])
    out = out_root / name
    out.mkdir(parents=True, exist_ok=True)
    frames.to_parquet(out / "frames.parquet", index=False)
    np.save(out / "windows_index.npy", windows)
    stats = {
        "dataset": name,
        "source_test_folder": test_folder,
        "window_size": window_size,
        "stride": stride,
        "split_counts": {str(k): int(v) for k, v in zip(*np.unique(windows[:, 4], return_counts=True))},
        "label_counts": {str(k): int(v) for k, v in zip(*np.unique(windows[:, 2], return_counts=True))},
    }
    (out / "ctt_prepare_meta.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"[ctt] wrote {out} frames={len(frames)} windows={len(windows)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--raw-root", default="data/raw/can-train-and-test")
    parser.add_argument("--datasets", nargs="+", default=list(TEST_FOLDERS.keys()))
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--stride", type=int, default=100)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    raw_root = (root / args.raw_root).resolve() if not Path(args.raw_root).is_absolute() else Path(args.raw_root)
    out_root = root / "data/processed"
    for name in args.datasets:
        prepare_one(raw_root, out_root, name, TEST_FOLDERS[name], args.window_size, args.stride, args.val_fraction)


if __name__ == "__main__":
    main()
