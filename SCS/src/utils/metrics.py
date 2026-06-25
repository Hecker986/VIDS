"""Classification metrics for experiment tables."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)


def find_best_threshold(y_true: np.ndarray, y_score: np.ndarray,
                        metric: str = "f1") -> float:
    thresholds = np.linspace(0.01, 0.99, 200)
    best_t, best_v = 0.5, -1.0
    for t in thresholds:
        yp = (y_score >= t).astype(int)
        if metric == "f1":
            v = float(f1_score(y_true, yp, zero_division=0))
        else:
            v = float(f1_score(y_true, yp, average="macro", zero_division=0))
        if v > best_v:
            best_v = v
            best_t = float(t)
    return best_t


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray,
                    threshold: float = 0.5) -> dict:
    y_pred = (y_score >= threshold).astype(int)
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "fpr": float(fp / max(fp + tn, 1)),
        "fnr": float(fn / max(fn + tp, 1)),
        "threshold": threshold,
    }
    if len(np.unique(y_true)) > 1:
        out["auroc"] = float(roc_auc_score(y_true, y_score))
    else:
        out["auroc"] = float("nan")
    return out
