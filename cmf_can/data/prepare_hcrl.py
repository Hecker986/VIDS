"""Prepare HCRL CAN-Intrusion and Car-Hacking datasets for CMF-CAN."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

SPLIT_TRAIN = 0
SPLIT_VAL = 1
SPLIT_TEST = 2

TXT_RE = re.compile(
    r"Timestamp:\s*(?P<timestamp>[0-9.]+)\s+ID:\s*(?P<can_id>[0-9A-Fa-f]+)"
    r"\s+\S+\s+DLC:\s*(?P<dlc>\d+)\s*(?P<payload>[0-9A-Fa-f ]*)"
)

FRAME_COLS = [
    "timestamp", "can_id", "dlc", "data0", "data1", "data2", "data3", "data4",
    "data5", "data6", "data7", "delta_t_global", "delta_t_same_id", "label",
    "attack_type", "dataset", "vehicle", "capture_id", "split_group",
]


def _split_for_positions(n: int, train_fraction: float, val_fraction: float) -> np.ndarray:
    train_end = int(n * train_fraction)
    val_end = train_end + int(n * val_fraction)
    split = np.full(n, SPLIT_TEST, dtype=np.int64)
    split[:train_end] = SPLIT_TRAIN
    split[train_end:val_end] = SPLIT_VAL
    return split


def _finish_capture(
    timestamp: np.ndarray,
    can_id: np.ndarray,
    dlc: np.ndarray,
    payload: np.ndarray,
    label: np.ndarray,
    attack_type: str,
    dataset: str,
    capture_id: str,
    train_fraction: float,
    val_fraction: float,
) -> pd.DataFrame:
    order = np.argsort(timestamp, kind="stable")
    timestamp = timestamp[order].astype(np.float64, copy=False)
    can_id = can_id[order].astype(np.int64, copy=False)
    dlc = dlc[order].astype(np.int64, copy=False)
    payload = payload[order].astype(np.uint8, copy=False)
    label = label[order].astype(np.int64, copy=False)
    n = len(timestamp)

    delta_global = np.zeros(n, dtype=np.float32)
    if n > 1:
        delta_global[1:] = np.maximum(np.diff(timestamp), 0.0).astype(np.float32)
    delta_same = (
        pd.Series(timestamp)
        .groupby(pd.Series(can_id), sort=False)
        .diff()
        .fillna(0.0)
        .clip(lower=0.0)
        .to_numpy(np.float32)
    )
    split = _split_for_positions(n, train_fraction, val_fraction)
    split_name = np.where(split == SPLIT_TRAIN, "train", np.where(split == SPLIT_VAL, "val", "test"))
    attack_labels = np.where(label == 1, attack_type, "normal")

    out = pd.DataFrame(
        {
            "timestamp": timestamp,
            "can_id": can_id,
            "dlc": dlc,
            "delta_t_global": delta_global,
            "delta_t_same_id": delta_same,
            "label": label,
            "attack_type": attack_labels,
            "dataset": dataset,
            "vehicle": "hcrl",
            "capture_id": capture_id,
            "split_group": [f"{capture_id}:{name}" for name in split_name],
        }
    )
    for i in range(8):
        out[f"data{i}"] = payload[:, i]
    return out[FRAME_COLS]


def _parse_txt_file(
    path: Path,
    label_value: int,
    attack_type: str,
    dataset: str,
    capture_id: str,
    train_fraction: float,
    val_fraction: float,
) -> pd.DataFrame:
    timestamps: list[float] = []
    ids: list[int] = []
    dlcs: list[int] = []
    payloads: list[list[int]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = TXT_RE.search(line)
            if not m:
                continue
            dlc = int(m.group("dlc"))
            bytes_text = m.group("payload").strip().split()
            row = [int(x, 16) for x in bytes_text[:8]]
            row.extend([0] * (8 - len(row)))
            timestamps.append(float(m.group("timestamp")))
            ids.append(int(m.group("can_id"), 16))
            dlcs.append(dlc)
            payloads.append(row)
    n = len(timestamps)
    labels = np.full(n, label_value, dtype=np.int64)
    return _finish_capture(
        np.asarray(timestamps, dtype=np.float64),
        np.asarray(ids, dtype=np.int64),
        np.asarray(dlcs, dtype=np.int64),
        np.asarray(payloads, dtype=np.uint8),
        labels,
        attack_type,
        dataset,
        capture_id,
        train_fraction,
        val_fraction,
    )


def _hex_series(series: pd.Series, dtype: type[np.integer] = np.uint8) -> np.ndarray:
    return np.fromiter((int(str(x).strip(), 16) for x in series), dtype=dtype, count=len(series))


def _parse_car_csv(
    path: Path,
    attack_type: str,
    train_fraction: float,
    val_fraction: float,
    chunksize: int,
) -> pd.DataFrame:
    names = ["timestamp", "can_id", "dlc", *[f"field{i}" for i in range(9)]]
    parts: list[pd.DataFrame] = []
    offset = 0
    for chunk in pd.read_csv(path, header=None, names=names, chunksize=chunksize):
        n = len(chunk)
        payload = np.zeros((n, 8), dtype=np.uint8)
        labels = np.zeros(n, dtype=np.int64)
        fields = chunk[[f"field{i}" for i in range(9)]].to_numpy(dtype=object, copy=False)
        for row_idx, row in enumerate(fields):
            values = [x for x in row if not pd.isna(x)]
            if not values:
                continue
            labels[row_idx] = int(str(values[-1]).strip() == "T")
            for byte_idx, value in enumerate(values[:-1][:8]):
                payload[row_idx, byte_idx] = int(str(value).strip(), 16)
        part = _finish_capture(
            chunk["timestamp"].to_numpy(np.float64, copy=True),
            _hex_series(chunk["can_id"], np.int64),
            chunk["dlc"].to_numpy(np.int64, copy=True),
            payload,
            labels,
            attack_type,
            "car_hacking",
            f"{path.stem}:{offset}",
            train_fraction,
            val_fraction,
        )
        parts.append(part)
        offset += n
        print(f"[hcrl] parsed {path.name} rows={offset}", flush=True)
    return pd.concat(parts, ignore_index=True)


def _build_windows(frames: pd.DataFrame, window_size: int, stride: int) -> np.ndarray:
    windows: list[list[int]] = []
    attack_type_codes, _ = pd.factorize(frames["attack_type"], sort=True)
    labels_all = frames["label"].to_numpy(np.int64)
    for _, idx in frames.groupby("split_group", sort=False).indices.items():
        arr = np.asarray(idx, dtype=np.int64)
        labels = labels_all[arr]
        if len(arr) < window_size:
            continue
        group_name = str(frames.iloc[arr[0]]["split_group"])
        split_id = SPLIT_TRAIN if group_name.endswith(":train") else SPLIT_VAL if group_name.endswith(":val") else SPLIT_TEST
        for off in range(0, len(arr) - window_size + 1, stride):
            win_idx = arr[off:off + window_size]
            win_labels = labels[off:off + window_size]
            y = int(win_labels.max())
            atk = int(attack_type_codes[win_idx][win_labels.astype(bool)][0]) if y else 0
            windows.append([int(win_idx[0]), int(win_idx[-1] + 1), y, atk, split_id])
    return np.asarray(windows, dtype=np.int64)


def _write_dataset(root: Path, name: str, frames: pd.DataFrame, window_size: int, stride: int, meta: dict) -> None:
    out = root / "data/processed" / name
    out.mkdir(parents=True, exist_ok=True)
    windows = _build_windows(frames, window_size, stride)
    frames.to_parquet(out / "frames.parquet", index=False)
    np.save(out / "windows_index.npy", windows)
    meta = {
        **meta,
        "dataset": name,
        "window_size": window_size,
        "stride": stride,
        "n_frames": int(len(frames)),
        "n_windows": int(len(windows)),
        "split_counts": {str(k): int(v) for k, v in zip(*np.unique(windows[:, 4], return_counts=True))},
        "label_counts": {str(k): int(v) for k, v in zip(*np.unique(windows[:, 2], return_counts=True))},
    }
    (out / "hcrl_prepare_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[hcrl] wrote {out} frames={len(frames)} windows={len(windows)}", flush=True)


def prepare_can_intrusion(root: Path, raw_root: Path, window_size: int, stride: int, train_fraction: float, val_fraction: float) -> None:
    files = [
        ("Attack_free_dataset.txt", 0, "normal"),
        ("DoS_attack_dataset.txt", 1, "dos"),
        ("Fuzzy_attack_dataset.txt", 1, "fuzzy"),
        ("Impersonation_attack_dataset.txt", 1, "impersonation"),
    ]
    parts = []
    for filename, label, attack_type in files:
        path = raw_root / filename
        parts.append(_parse_txt_file(path, label, attack_type, "hcrl_can_intrusion", path.stem, train_fraction, val_fraction))
        print(f"[hcrl] parsed {path}", flush=True)
    frames = pd.concat(parts, ignore_index=True)
    _write_dataset(
        root,
        "hcrl_can_intrusion",
        frames,
        window_size,
        stride,
        {"source": "HCRL CAN-Intrusion/OTIDS Dropbox", "frame_label_note": "attack files do not include per-frame labels; all frames in attack captures are marked anomalous"},
    )


def prepare_car_hacking(root: Path, raw_root: Path, window_size: int, stride: int, train_fraction: float, val_fraction: float, chunksize: int) -> None:
    files = [
        ("DoS_dataset.csv", "dos"),
        ("Fuzzy_dataset.csv", "fuzzy"),
        ("RPM_dataset.csv", "rpm"),
        ("gear_dataset.csv", "gear"),
    ]
    parts = []
    normal_path = raw_root / "normal_run_data" / "normal_run_data.txt"
    parts.append(_parse_txt_file(normal_path, 0, "normal", "car_hacking", "normal_run_data", train_fraction, val_fraction))
    for filename, attack_type in files:
        parts.append(_parse_car_csv(raw_root / filename, attack_type, train_fraction, val_fraction, chunksize))
    frames = pd.concat(parts, ignore_index=True)
    _write_dataset(
        root,
        "car_hacking",
        frames,
        window_size,
        stride,
        {"source": "HCRL Car-Hacking Dropbox", "frame_label_note": "CSV T flag is anomalous; R flag and normal_run_data are benign"},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--raw-can-intrusion", default="data/raw/hcrl_can_intrusion/extracted")
    parser.add_argument("--raw-car-hacking", default="data/raw/car_hacking/extracted")
    parser.add_argument("--datasets", nargs="+", default=["hcrl_can_intrusion", "car_hacking"])
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--stride", type=int, default=100)
    parser.add_argument("--train-fraction", type=float, default=0.6)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--chunksize", type=int, default=500_000)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    if "hcrl_can_intrusion" in args.datasets:
        prepare_can_intrusion(root, root / args.raw_can_intrusion, args.window_size, args.stride, args.train_fraction, args.val_fraction)
    if "car_hacking" in args.datasets:
        prepare_car_hacking(root, root / args.raw_car_hacking, args.window_size, args.stride, args.train_fraction, args.val_fraction, args.chunksize)


if __name__ == "__main__":
    main()
