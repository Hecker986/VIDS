"""Metrics required by the CMF-CAN plan."""
from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _confusion(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    return tn, fp, fn, tp


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=np.float64)
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = _confusion(y_true, y_pred)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "aupr": float("nan"),
        "auroc": float("nan"),
        "fpr": float(fp / max(fp + tn, 1)),
        "fnr": float(fn / max(fn + tp, 1)),
        "threshold": float(threshold),
    }
    if len(np.unique(y_true)) > 1:
        out["auroc"] = float(roc_auc_score(y_true, y_score))
        out["aupr"] = float(average_precision_score(y_true, y_score))
    return out


def best_threshold(y_true: np.ndarray, y_score: np.ndarray, metric: str = "f1") -> float:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=np.float64)
    thresholds = np.unique(np.quantile(y_score, np.linspace(0.0, 1.0, 401)))
    thresholds = np.concatenate(([0.0], thresholds, [1.0]))
    best_t, best_v = 0.5, -math.inf
    pos_total = max(int((y_true == 1).sum()), 1)
    neg_total = max(int((y_true == 0).sum()), 1)
    for t in thresholds:
        pred = (y_score >= t).astype(int)
        tp = int(((y_true == 1) & (pred == 1)).sum())
        fp = int(((y_true == 0) & (pred == 1)).sum())
        fn = pos_total - tp
        tn = neg_total - fp
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1_pos = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        if metric == "macro_f1":
            precision_0 = tn / max(tn + fn, 1)
            recall_0 = tn / max(tn + fp, 1)
            f1_neg = 0.0 if precision_0 + recall_0 == 0 else 2 * precision_0 * recall_0 / (precision_0 + recall_0)
            value = 0.5 * (f1_pos + f1_neg)
        else:
            value = f1_pos
        if value > best_v:
            best_t, best_v = float(t), float(value)
    return best_t


def constrained_fpr_metrics(y_true: np.ndarray, y_score: np.ndarray, limits=(1e-4, 5e-4, 1e-3)) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=np.float64)
    out: dict[str, float] = {}
    order = np.argsort(-y_score)
    sorted_scores = y_score[order]
    sorted_true = y_true[order]
    pos_total = max(int((y_true == 1).sum()), 1)
    neg_total = max(int((y_true == 0).sum()), 1)
    unique_end = np.r_[np.where(sorted_scores[:-1] != sorted_scores[1:])[0], len(sorted_scores) - 1]
    tp = np.cumsum(sorted_true == 1)[unique_end].astype(np.float64)
    fp = np.cumsum(sorted_true == 0)[unique_end].astype(np.float64)
    thresholds = sorted_scores[unique_end]
    recall = tp / pos_total
    fpr = fp / neg_total
    precision = tp / np.maximum(tp + fp, 1.0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(denom), where=denom > 0)
    for limit in limits:
        key = f"{limit:.0e}".replace("-", "m")
        valid = np.where(fpr <= limit)[0]
        if len(valid) == 0:
            out[f"recall_at_fpr_{key}"] = 0.0
            out[f"f1_at_fpr_{key}"] = 0.0
            out[f"actual_fpr_at_{key}"] = 0.0
            out[f"threshold_at_fpr_{key}"] = 1.0
        else:
            best_local = valid[np.lexsort((f1[valid], recall[valid]))[-1]]
            out[f"recall_at_fpr_{key}"] = float(recall[best_local])
            out[f"f1_at_fpr_{key}"] = float(f1[best_local])
            out[f"actual_fpr_at_{key}"] = float(fpr[best_local])
            out[f"threshold_at_fpr_{key}"] = float(thresholds[best_local])
    return out
