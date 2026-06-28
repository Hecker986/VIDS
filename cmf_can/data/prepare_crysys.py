"""Prepare a reproducible CrySyS subset directly from the official Figshare zip."""
from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

SPLIT_TRAIN = 0
SPLIT_VAL = 1
SPLIT_TEST = 2

LOG_RE = re.compile(r"\((?P<timestamp>[0-9.]+)\)\s+\S+\s+(?P<can_id>[0-9A-Fa-f]+)#(?P<payload>[0-9A-Fa-f]*)")

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


def _json_name(log_name: str) -> str:
    return f"{log_name[:-4]}.json"


def _scenario_name(path: str) -> str:
    parts = path.split("/")
    return parts[-2] if len(parts) >= 2 else "unknown"


def _attack_family(path: str) -> str:
    name = Path(path).stem
    if "-malicious-" not in name:
        return "normal"
    for family in ("ADD-DECR", "ADD-INCR", "CONST", "NEG-OFFSET", "NEG_OFFSET", "POS-OFFSET", "POS_OFFSET", "REPLAY"):
        if family in name:
            return family.replace("-", "_").lower()
    return "other"


def _attack_type(path: str) -> str:
    name = Path(path).stem
    if "-malicious-" not in name:
        return "normal"
    text = name.split("-malicious-", 1)[1]
    text = text.split("-0x", 1)[0]
    return text.replace("-", "_").lower()


def _intervals(meta: dict) -> list[tuple[float, float]]:
    starts: list[float] = []
    intervals: list[tuple[float, float]] = []
    for marker in meta.get("markers", []):
        desc = str(marker.get("description", "")).lower()
        t = float(marker.get("time", 0.0))
        if "start" in desc:
            starts.append(t)
        elif "end" in desc and starts:
            intervals.append((starts.pop(0), t))
    return intervals


def _select_logs_by_count(zf: zipfile.ZipFile, malicious_per_scenario: int) -> list[str]:
    logs = [n for n in zf.namelist() if n.endswith(".log") and "inj-messages.log" not in n]
    benign = sorted(n for n in logs if n.endswith("-benign.log"))
    by_scenario: dict[str, list[str]] = {}
    for name in sorted(n for n in logs if "-malicious-" in n):
        by_scenario.setdefault(_scenario_name(name), []).append(name)

    selected = list(benign)
    for scenario in sorted(by_scenario):
        candidates = by_scenario[scenario]
        picked: list[str] = []
        for keyword in ("msg-inj", "msg-mod"):
            for name in candidates:
                if keyword in name and name not in picked:
                    picked.append(name)
                    break
        for name in candidates:
            if len(picked) >= malicious_per_scenario:
                break
            if name not in picked:
                picked.append(name)
        selected.extend(picked[:malicious_per_scenario])
    return selected


def _select_logs_by_family(zf: zipfile.ZipFile, families: list[str], attack_mode: str) -> list[str]:
    logs = [n for n in zf.namelist() if n.endswith(".log") and "inj-messages.log" not in n]
    benign = sorted(n for n in logs if n.endswith("-benign.log"))
    by_scenario: dict[str, list[str]] = {}
    for name in sorted(n for n in logs if "-malicious-" in n):
        by_scenario.setdefault(_scenario_name(name), []).append(name)

    selected = list(benign)
    wanted = {family.replace("-", "_").lower() for family in families}
    for scenario in sorted(by_scenario):
        candidates = by_scenario[scenario]
        for family in sorted(wanted):
            family_candidates = [name for name in candidates if _attack_family(name) == family]
            if not family_candidates:
                continue
            if attack_mode == "inj":
                preferred = [name for name in family_candidates if "msg-inj" in name]
            elif attack_mode == "mod":
                preferred = [name for name in family_candidates if "msg-mod" in name]
            else:
                preferred = [name for name in family_candidates if "msg-inj" in name]
            selected.append((preferred or family_candidates)[0])
    return selected


def _parse_capture(
    zf: zipfile.ZipFile,
    log_name: str,
    dataset_name: str,
) -> pd.DataFrame:
    with zf.open(_json_name(log_name)) as f:
        meta = json.load(f)
    intervals = _intervals(meta)
    malicious = str(meta.get("label", "")).lower() == "malicious"
    attack_type = _attack_type(log_name)
    capture_id = Path(log_name).stem
    scenario = _scenario_name(log_name)

    ts: list[float] = []
    ids: list[int] = []
    dlc: list[int] = []
    payloads: list[list[int]] = []
    labels: list[int] = []
    with zf.open(log_name) as f:
        for raw in f:
            line = raw.decode("utf-8", errors="ignore")
            m = LOG_RE.search(line)
            if not m:
                continue
            t = float(m.group("timestamp"))
            payload_hex = m.group("payload")
            row = [int(payload_hex[i:i + 2], 16) for i in range(0, min(len(payload_hex), 16), 2)]
            row.extend([0] * (8 - len(row)))
            is_attack = malicious and any(start <= t <= end for start, end in intervals)
            ts.append(t)
            ids.append(int(m.group("can_id"), 16))
            dlc.append(min(len(payload_hex) // 2, 8))
            payloads.append(row)
            labels.append(int(is_attack))

    timestamp = np.asarray(ts, dtype=np.float64)
    order = np.argsort(timestamp, kind="stable")
    timestamp = timestamp[order]
    can_id = np.asarray(ids, dtype=np.int64)[order]
    dlc_arr = np.asarray(dlc, dtype=np.int64)[order]
    payload = np.asarray(payloads, dtype=np.uint8)[order]
    label = np.asarray(labels, dtype=np.int64)[order]
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
    out = pd.DataFrame(
        {
            "timestamp": timestamp,
            "can_id": can_id,
            "dlc": dlc_arr,
            "delta_t_global": delta_global,
            "delta_t_same_id": delta_same,
            "label": label,
            "attack_type": np.where(label == 1, attack_type, "normal"),
            "dataset": dataset_name,
            "vehicle": scenario,
            "capture_id": capture_id,
            "split_group": capture_id,
        }
    )
    for i in range(8):
        out[f"data{i}"] = payload[:, i]
    return out[FRAME_COLS]


def _build_windows(
    frames: pd.DataFrame,
    window_size: int,
    stride: int,
    train_fraction: float,
    val_fraction: float,
    seed: int,
) -> np.ndarray:
    windows: list[list[int]] = []
    attack_type_codes, _ = pd.factorize(frames["attack_type"], sort=True)
    labels_all = frames["label"].to_numpy(np.int64)
    for _, idx in frames.groupby("split_group", sort=False).indices.items():
        arr = np.asarray(idx, dtype=np.int64)
        if len(arr) < window_size:
            continue
        labels = labels_all[arr]
        for off in range(0, len(arr) - window_size + 1, stride):
            win_idx = arr[off:off + window_size]
            win_labels = labels[off:off + window_size]
            y = int(win_labels.max())
            atk = int(attack_type_codes[win_idx][win_labels.astype(bool)][0]) if y else 0
            windows.append([int(win_idx[0]), int(win_idx[-1] + 1), y, atk, SPLIT_TRAIN])
    out = np.asarray(windows, dtype=np.int64)
    rng = np.random.default_rng(seed)
    for cls in (0, 1):
        idx = np.where(out[:, 2] == cls)[0]
        rng.shuffle(idx)
        train_end = int(len(idx) * train_fraction)
        val_end = train_end + int(len(idx) * val_fraction)
        out[idx[:train_end], 4] = SPLIT_TRAIN
        out[idx[train_end:val_end], 4] = SPLIT_VAL
        out[idx[val_end:], 4] = SPLIT_TEST
    return out


def prepare_crysys(
    root: Path,
    zip_path: Path,
    dataset_name: str,
    selection: str,
    malicious_per_scenario: int,
    families: list[str],
    attack_mode: str,
    window_size: int,
    stride: int,
    train_fraction: float,
    val_fraction: float,
    seed: int,
) -> None:
    parts: list[pd.DataFrame] = []
    with zipfile.ZipFile(zip_path) as zf:
        if selection == "family":
            selected = _select_logs_by_family(zf, families, attack_mode)
        else:
            selected = _select_logs_by_count(zf, malicious_per_scenario)
        for i, log_name in enumerate(selected, 1):
            parts.append(_parse_capture(zf, log_name, dataset_name))
            print(f"[crysys] parsed {i}/{len(selected)} {log_name}", flush=True)
    frames = pd.concat(parts, ignore_index=True)
    windows = _build_windows(frames, window_size, stride, train_fraction, val_fraction, seed)

    out = root / "data/processed" / dataset_name
    out.mkdir(parents=True, exist_ok=True)
    frames.to_parquet(out / "frames.parquet", index=False)
    np.save(out / "windows_index.npy", windows)
    meta = {
        "dataset": dataset_name,
        "source": "Figshare 10.6084/m9.figshare.23624208.v1",
        "zip_path": str(zip_path),
        "selection": selection,
        "attack_mode": attack_mode,
        "malicious_per_scenario": malicious_per_scenario,
        "families": families,
        "selected_captures": [str(x) for x in frames["capture_id"].drop_duplicates().tolist()],
        "window_size": window_size,
        "stride": stride,
        "split_method": "stratified non-overlapping windows after per-capture windowing",
        "seed": seed,
        "n_frames": int(len(frames)),
        "n_windows": int(len(windows)),
        "split_counts": {str(k): int(v) for k, v in zip(*np.unique(windows[:, 4], return_counts=True))},
        "label_counts": {str(k): int(v) for k, v in zip(*np.unique(windows[:, 2], return_counts=True))},
        "label_note": "malicious frames are labeled from JSON attack start/end markers; benign captures are labeled normal",
    }
    (out / "crysys_prepare_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[crysys] wrote {out} frames={len(frames)} windows={len(windows)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--zip-path", default="data/raw/crysys/crysys_dataset.zip")
    parser.add_argument("--dataset-name", default="crysys_subset")
    parser.add_argument("--selection", choices=["count", "family"], default="count")
    parser.add_argument("--attack-mode", choices=["any", "inj", "mod"], default="any")
    parser.add_argument("--malicious-per-scenario", type=int, default=2)
    parser.add_argument(
        "--families",
        nargs="+",
        default=["ADD-DECR", "ADD-INCR", "CONST", "NEG-OFFSET", "POS-OFFSET", "REPLAY"],
    )
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--stride", type=int, default=100)
    parser.add_argument("--train-fraction", type=float, default=0.6)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    prepare_crysys(
        root,
        root / args.zip_path,
        args.dataset_name,
        args.selection,
        args.malicious_per_scenario,
        args.families,
        args.attack_mode,
        args.window_size,
        args.stride,
        args.train_fraction,
        args.val_fraction,
        args.seed,
    )


if __name__ == "__main__":
    main()
