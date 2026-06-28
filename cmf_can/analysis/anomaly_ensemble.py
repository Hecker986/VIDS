from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from torch.utils.data import DataLoader

from cmf_can.analysis.calibrate import collect_logits, evaluate_method, softmax_score
from cmf_can.data.collate import collate_batch
from cmf_can.data.dataset import CMFWindowDataset, SPLIT_TEST, SPLIT_TRAIN, SPLIT_VAL
from cmf_can.models.cmf import build_model


def _labels(ds: CMFWindowDataset) -> np.ndarray:
    return ds.windows[:, 2].astype(np.int64)


def _stats(ds: CMFWindowDataset) -> np.ndarray:
    return np.asarray(ds.window_stats[ds.row_indices], dtype=np.float32)


def _rank01(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    return ranks / max(len(x) - 1, 1)


def _minmax_train(val: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lo = float(np.min(val))
    hi = float(np.max(val))
    scale = max(hi - lo, 1e-12)
    return np.clip((val - lo) / scale, 0.0, 1.0), np.clip((test - lo) / scale, 0.0, 1.0)


def _causal_rolling_mean(scores: np.ndarray, window: int) -> np.ndarray:
    out = np.empty_like(scores, dtype=np.float64)
    csum = np.cumsum(np.r_[0.0, scores.astype(np.float64)])
    for i in range(len(scores)):
        start = max(0, i + 1 - window)
        out[i] = (csum[i + 1] - csum[start]) / (i + 1 - start)
    return out


def _causal_rolling_max(scores: np.ndarray, window: int) -> np.ndarray:
    out = np.empty_like(scores, dtype=np.float64)
    values = scores.astype(np.float64)
    for i in range(len(values)):
        start = max(0, i + 1 - window)
        out[i] = values[start:i + 1].max()
    return out


def add_temporal_scores(candidates: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    out = dict(candidates)
    for name, (val_s, test_s) in candidates.items():
        if name.startswith("ensemble_"):
            continue
        for window in (3, 5, 9):
            out[f"{name}_rollmean{window}"] = (
                _causal_rolling_mean(val_s, window),
                _causal_rolling_mean(test_s, window),
            )
            out[f"{name}_rollmax{window}"] = (
                _causal_rolling_max(val_s, window),
                _causal_rolling_max(test_s, window),
            )
    return out


def robust_scores(train_normal: np.ndarray, val_x: np.ndarray, test_x: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    median = np.median(train_normal, axis=0)
    q25, q75 = np.percentile(train_normal, [25, 75], axis=0)
    iqr = np.maximum(q75 - q25, 1e-6)
    mean = train_normal.mean(axis=0)
    std = np.maximum(train_normal.std(axis=0), 1e-6)

    def robust_mean(x: np.ndarray) -> np.ndarray:
        return np.abs((x - median) / iqr).mean(axis=1)

    def robust_max(x: np.ndarray) -> np.ndarray:
        return np.abs((x - median) / iqr).max(axis=1)

    def diag_mahal(x: np.ndarray) -> np.ndarray:
        return (((x - mean) / std) ** 2).mean(axis=1)

    lo = np.percentile(train_normal, 0.5, axis=0)
    hi = np.percentile(train_normal, 99.5, axis=0)

    def tail_count(x: np.ndarray) -> np.ndarray:
        return ((x < lo) | (x > hi)).mean(axis=1)

    return {
        "stats_robust_mean": (robust_mean(val_x), robust_mean(test_x)),
        "stats_robust_max": (robust_max(val_x), robust_max(test_x)),
        "stats_diag_mahal": (diag_mahal(val_x), diag_mahal(test_x)),
        "stats_tail_count": (tail_count(val_x), tail_count(test_x)),
    }


def covariance_scores(train_normal: np.ndarray, val_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    clf = LedoitWolf().fit(train_normal)
    return clf.mahalanobis(val_x), clf.mahalanobis(test_x)


def pca_scores(train_normal: np.ndarray, val_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_components = min(12, train_normal.shape[1], max(1, train_normal.shape[0] - 1))
    pca = PCA(n_components=n_components, random_state=0).fit(train_normal)

    def reconstruction_error(x: np.ndarray) -> np.ndarray:
        z = pca.transform(x)
        x_hat = pca.inverse_transform(z)
        return ((x - x_hat) ** 2).mean(axis=1)

    return reconstruction_error(val_x), reconstruction_error(test_x)


def isolation_scores(train_normal: np.ndarray, val_x: np.ndarray, test_x: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    max_samples = min(len(train_normal), 10000)
    clf = IsolationForest(
        n_estimators=300,
        max_samples=max_samples,
        contamination="auto",
        random_state=seed,
        n_jobs=-1,
    )
    clf.fit(train_normal)
    return -clf.score_samples(val_x), -clf.score_samples(test_x)


@torch.no_grad()
def model_scores(root: Path, dataset: str, model_name: str, seed: int, label_ratio: float, batch_size: int, num_workers: int) -> tuple[np.ndarray, np.ndarray] | None:
    ckpt = root / "checkpoints/cmf" / dataset / model_name / f"ratio_{label_ratio:g}_seed_{seed}" / "best.pt"
    if not ckpt.exists():
        return None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_name).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    val_ds = CMFWindowDataset(root, dataset, SPLIT_VAL)
    test_ds = CMFWindowDataset(root, dataset, SPLIT_TEST)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch, num_workers=num_workers, pin_memory=device.type == "cuda")
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch, num_workers=num_workers, pin_memory=device.type == "cuda")
    _, val_logits = collect_logits(model, val_loader, device)
    _, test_logits = collect_logits(model, test_loader, device)
    return softmax_score(val_logits), softmax_score(test_logits)


def run(
    root: Path,
    dataset: str,
    model_name: str,
    seed: int,
    label_ratio: float,
    batch_size: int,
    num_workers: int,
    table: str,
) -> list[dict]:
    train_ds = CMFWindowDataset(root, dataset, SPLIT_TRAIN)
    val_ds = CMFWindowDataset(root, dataset, SPLIT_VAL)
    test_ds = CMFWindowDataset(root, dataset, SPLIT_TEST)
    train_x = _stats(train_ds)
    train_y = _labels(train_ds)
    normal_x = train_x[train_y == 0]
    val_x = _stats(val_ds)
    test_x = _stats(test_ds)
    val_y = _labels(val_ds)
    test_y = _labels(test_ds)

    candidates = robust_scores(normal_x, val_x, test_x)
    candidates["stats_ledoit_mahal"] = covariance_scores(normal_x, val_x, test_x)
    candidates["stats_pca_recon"] = pca_scores(normal_x, val_x, test_x)
    candidates["isolation_forest"] = isolation_scores(normal_x, val_x, test_x, seed)
    candidates = add_temporal_scores(candidates)
    net_scores = model_scores(root, dataset, model_name, seed, label_ratio, batch_size, num_workers)
    if net_scores is not None:
        candidates["model_attack_prob"] = net_scores
        candidates.update(add_temporal_scores({"model_attack_prob": net_scores}))
        for name, (val_s, test_s) in list(candidates.items()):
            if name == "model_attack_prob":
                continue
            for alpha in (0.25, 0.5, 0.75):
                candidates[f"ensemble_a{alpha:g}_{name}"] = (
                    alpha * _rank01(net_scores[0]) + (1.0 - alpha) * _rank01(val_s),
                    alpha * _rank01(net_scores[1]) + (1.0 - alpha) * _rank01(test_s),
                )

    rows: list[dict] = []
    for score_name, (val_s_raw, test_s_raw) in candidates.items():
        val_s, test_s = _minmax_train(val_s_raw, test_s_raw)
        for policy in ["val_f1", "val_fpr_1em04", "val_fpr_5em04", "val_fpr_1em03"]:
            rows.append({
                "dataset": dataset,
                "model": model_name,
                "seed": seed,
                "label_ratio": label_ratio,
                "score": score_name,
                **evaluate_method(score_name, val_y, val_s, test_y, test_s, policy),
            })

    out = root / "results/cmf_tables" / table
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    write_header = not out.exists()
    with out.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", default="cmf_can")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label-ratio", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--table", default="anomaly_ensemble_trials.csv")
    args = parser.parse_args()
    rows = run(Path(args.root).resolve(), args.dataset, args.model, args.seed, args.label_ratio, args.batch_size, args.num_workers, args.table)
    best = max(rows, key=lambda row: row["f1"])
    low_fpr = max(rows, key=lambda row: row["recall_at_fpr_1em04"])
    print(
        f"[anomaly_ensemble] {args.dataset}/{args.model} "
        f"best_f1={best['score']}/{best['threshold_policy']} f1={best['f1']:.4f} "
        f"aupr={best['aupr']:.4f} fpr={best['fpr']:.4f}; "
        f"best_lowfpr={low_fpr['score']} recall@1e-4={low_fpr['recall_at_fpr_1em04']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
