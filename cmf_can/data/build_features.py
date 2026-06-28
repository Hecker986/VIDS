"""Build CMF-CAN three-modality features from a processed CAN dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SPLIT_TRAIN = 0
ID_SIZE = 4098


def _entropy(values: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    _, counts = np.unique(values, return_counts=True)
    p = counts.astype(np.float64) / counts.sum()
    return float(-(p * np.log2(p + 1e-12)).sum())


def _safe_standardize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / np.where(std < 1e-6, 1.0, std)


def _train_frame_mask(n_frames: int, windows: np.ndarray) -> np.ndarray:
    mask = np.zeros(n_frames, dtype=bool)
    for start, end, _, _, split in windows:
        if int(split) == SPLIT_TRAIN:
            mask[int(start):int(end)] = True
    return mask


def _compute_context(
    cid: np.ndarray,
    payload: np.ndarray,
    dlc: np.ndarray,
    dt_same: np.ndarray,
    train_mask: np.ndarray,
    trans_counts: np.ndarray,
) -> tuple[np.ndarray, dict]:
    context = np.zeros((ID_SIZE, 18), dtype=np.float32)
    train_ids = cid[train_mask]
    total_train = max(int(train_mask.sum()), 1)
    counts = np.bincount(train_ids, minlength=ID_SIZE).astype(np.float64)
    payload_train = payload[train_mask].astype(np.float32)
    dlc_train = dlc[train_mask].astype(np.float32)
    dt_train = dt_same[train_mask].astype(np.float32)

    global_payload_mean = payload_train.mean(axis=0) if len(payload_train) else np.zeros(8, dtype=np.float32)
    global_payload_std = payload_train.std(axis=0) if len(payload_train) else np.ones(8, dtype=np.float32)
    global_dlc_mean = float(dlc_train.mean()) if len(dlc_train) else 0.0
    global_period_mean = float(dt_train.mean()) if len(dt_train) else 0.0
    global_period_std = float(dt_train.std() + 1e-6) if len(dt_train) else 1.0

    for can_id in np.unique(train_ids):
        idx = train_mask & (cid == int(can_id))
        if not idx.any():
            continue
        pay = payload[idx].astype(np.float32)
        periods = dt_same[idx].astype(np.float32)
        row = trans_counts[int(can_id)].astype(np.float64)
        row_sum = row.sum()
        probs = row / row_sum if row_sum > 0 else np.zeros_like(row)
        nz = probs[probs > 0]
        trans_entropy = float(-(nz * np.log2(nz + 1e-12)).sum()) if len(nz) else 0.0
        rare_ratio = float((probs[probs > 0] < 0.01).sum() / max((probs > 0).sum(), 1))
        context[int(can_id), 0] = np.log1p(counts[int(can_id)])
        context[int(can_id), 1] = counts[int(can_id)] / total_train
        context[int(can_id), 2] = float(periods.mean())
        context[int(can_id), 3] = float(periods.std())
        context[int(can_id), 4] = float(dlc[idx].mean())
        context[int(can_id), 5:13] = pay.mean(axis=0)
        context[int(can_id), 13] = float(np.abs(np.diff(pay, axis=0)).sum(axis=1).mean()) if len(pay) > 1 else 0.0
        context[int(can_id), 14] = float((row > 0).sum())
        context[int(can_id), 15] = trans_entropy
        context[int(can_id), 16] = rare_ratio
        context[int(can_id), 17] = float(row.max() / row_sum) if row_sum > 0 else 0.0

    raw = context.copy()
    nonzero = counts > 0
    mean = raw[nonzero].mean(axis=0) if nonzero.any() else np.zeros(raw.shape[1], dtype=np.float32)
    std = raw[nonzero].std(axis=0) if nonzero.any() else np.ones(raw.shape[1], dtype=np.float32)
    context = _safe_standardize(raw, mean, std).astype(np.float32)
    context[~nonzero] = 0.0
    meta = {
        "id_context_features": [
            "log_count", "ratio", "period_mean", "period_std", "dlc_mean",
            "payload_mean_0", "payload_mean_1", "payload_mean_2", "payload_mean_3",
            "payload_mean_4", "payload_mean_5", "payload_mean_6", "payload_mean_7",
            "payload_change_rate", "transition_out_degree", "transition_entropy",
            "rare_successor_ratio", "max_successor_prob",
        ],
        "context_mean": mean.tolist(),
        "context_std": np.where(std < 1e-6, 1.0, std).tolist(),
        "global_payload_mean": global_payload_mean.tolist(),
        "global_payload_std": np.where(global_payload_std < 1e-6, 1.0, global_payload_std).tolist(),
        "global_dlc_mean": global_dlc_mean,
        "global_period_mean": global_period_mean,
        "global_period_std": global_period_std,
    }
    return context, meta


def build_features(root: Path, dataset: str, topk: int = 5, force: bool = False) -> Path:
    proc = root / "data/processed" / dataset
    out_dir = proc / "cmf_features"
    out_dir.mkdir(parents=True, exist_ok=True)
    done = out_dir / "feature_meta.json"
    if done.exists() and not force:
        print(f"[cmf-features] reuse {out_dir}", flush=True)
        return out_dir

    frames = pd.read_parquet(proc / "frames.parquet")
    windows = np.load(proc / "windows_index.npy")
    cid = frames["can_id"].to_numpy(np.int64, copy=True).clip(0, ID_SIZE - 1)
    payload = np.stack([frames[f"data{i}"].to_numpy(np.uint8, copy=True) for i in range(8)], axis=1)
    dlc = frames["dlc"].to_numpy(np.float32, copy=True)
    dt_global = frames["delta_t_global"].to_numpy(np.float32, copy=True)
    dt_same = frames["delta_t_same_id"].to_numpy(np.float32, copy=True)
    groups = frames["split_group"].astype("category").cat.codes.to_numpy(np.int64, copy=True)
    train_mask = _train_frame_mask(len(frames), windows)

    valid_transition = train_mask[:-1] & train_mask[1:] & (groups[:-1] == groups[1:])
    trans_counts = np.zeros((ID_SIZE, ID_SIZE), dtype=np.uint32)
    np.add.at(trans_counts, (cid[:-1][valid_transition], cid[1:][valid_transition]), 1)
    row_sum = trans_counts.sum(axis=1).astype(np.float64)
    top_next = np.argsort(-trans_counts, axis=1)[:, :topk]

    id_context, context_meta = _compute_context(cid, payload, dlc, dt_same, train_mask, trans_counts)

    train_dt = np.stack([dt_global[train_mask], dt_same[train_mask]], axis=1)
    time_mean = train_dt.mean(axis=0)
    time_std = np.where(train_dt.std(axis=0) < 1e-6, 1.0, train_dt.std(axis=0))
    period_mean_raw = context_meta["global_period_mean"]
    period_std_raw = context_meta["global_period_std"]

    trans_prob = np.zeros(len(frames), dtype=np.float32)
    top_hit = np.zeros(len(frames), dtype=np.float32)
    same_group = groups[:-1] == groups[1:]
    cur = cid[:-1][same_group]
    nxt = cid[1:][same_group]
    denom = np.maximum(row_sum[cur], 1.0)
    trans_prob[:-1][same_group] = trans_counts[cur, nxt] / denom
    top_hit[:-1][same_group] = (top_next[cur] == nxt[:, None]).any(axis=1).astype(np.float32)
    trans_rarity = -np.log(trans_prob + 1e-8).astype(np.float32)

    payload_delta_l1 = np.zeros(len(frames), dtype=np.float32)
    payload_delta_l2 = np.zeros(len(frames), dtype=np.float32)
    payload_changed = np.zeros(len(frames), dtype=np.float32)
    last_payload = np.full((ID_SIZE, 8), -1, dtype=np.int16)
    for i in range(len(frames)):
        c = cid[i]
        prev = last_payload[c]
        if prev[0] >= 0:
            diff = payload[i].astype(np.int16) - prev
            payload_delta_l1[i] = float(np.abs(diff).sum()) / (8.0 * 255.0)
            payload_delta_l2[i] = float(np.sqrt((diff.astype(np.float32) ** 2).sum())) / (np.sqrt(8.0) * 255.0)
            payload_changed[i] = float(np.any(diff != 0))
        last_payload[c] = payload[i].astype(np.int16)
        if i and i % 5_000_000 == 0:
            print(f"[cmf-features] payload delta {i}/{len(frames)}", flush=True)

    frame_numeric = np.stack(
        [
            dlc / 8.0,
            (dt_global - time_mean[0]) / time_std[0],
            (dt_same - time_mean[1]) / time_std[1],
            (dt_same - period_mean_raw) / period_std_raw,
            payload_delta_l1,
            payload_delta_l2,
            payload_changed,
            trans_prob,
            trans_rarity,
            top_hit,
        ],
        axis=1,
    ).astype(np.float32)
    np.save(out_dir / "frame_numeric.npy", frame_numeric)
    np.save(out_dir / "id_context.npy", id_context)

    stats_rows = np.zeros((len(windows), 26), dtype=np.float32)
    for i, (start, end, _, _, _) in enumerate(windows):
        s, e = int(start), int(end)
        ids = cid[s:e]
        pay = payload[s:e].astype(np.float32) / 255.0
        nums = frame_numeric[s:e]
        probs = trans_prob[s:e]
        stats_rows[i] = np.array(
            [
                _entropy(ids),
                len(np.unique(ids)) / max(e - s, 1),
                np.bincount(ids, minlength=ID_SIZE).max() / max(e - s, 1),
                pay.mean(),
                pay.std(),
                pay.min(),
                pay.max(),
                dlc[s:e].mean() / 8.0,
                dlc[s:e].std() / 8.0,
                dt_global[s:e].mean(),
                dt_global[s:e].std(),
                dt_same[s:e].mean(),
                dt_same[s:e].std(),
                nums[:, 4].mean(),
                nums[:, 4].std(),
                nums[:, 6].mean(),
                probs.mean(),
                probs.std(),
                trans_rarity[s:e].mean(),
                trans_rarity[s:e].std(),
                top_hit[s:e].mean(),
                nums[:, 1].mean(),
                nums[:, 2].mean(),
                nums[:, 7].max(),
                nums[:, 8].max(),
                nums[:, 9].sum() / max(e - s, 1),
            ],
            dtype=np.float32,
        )
        if i and i % 50_000 == 0:
            print(f"[cmf-features] window stats {i}/{len(windows)}", flush=True)

    train_windows = windows[:, 4] == SPLIT_TRAIN
    stat_mean = stats_rows[train_windows].mean(axis=0)
    stat_std = np.where(stats_rows[train_windows].std(axis=0) < 1e-6, 1.0, stats_rows[train_windows].std(axis=0))
    stats_rows = _safe_standardize(stats_rows, stat_mean, stat_std).astype(np.float32)
    np.save(out_dir / "window_stats.npy", stats_rows)

    meta = {
        "dataset": dataset,
        "n_frames": int(len(frames)),
        "n_windows": int(len(windows)),
        "train_only_statistics": True,
        "frame_numeric_features": [
            "dlc_norm", "delta_t_global_z", "delta_t_same_id_z", "period_zscore",
            "payload_delta_l1", "payload_delta_l2", "payload_change_flag",
            "transition_prob", "transition_rarity", "topk_successor_hit",
        ],
        "window_stats_dim": int(stats_rows.shape[1]),
        "window_stats_mean": stat_mean.tolist(),
        "window_stats_std": stat_std.tolist(),
        "time_mean": time_mean.tolist(),
        "time_std": time_std.tolist(),
        **context_meta,
    }
    done.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[cmf-features] wrote {out_dir}", flush=True)
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    build_features(Path(args.root).resolve(), args.dataset, topk=args.topk, force=args.force)


if __name__ == "__main__":
    main()

