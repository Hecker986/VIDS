"""Attack-centric evaluation helpers for rare-event CAN IDS.

The functions in this module intentionally report attack-positive and
normal-positive metrics side by side. In highly imbalanced CAN IDS settings,
accuracy and weighted averages can be dominated by normal traffic and can rank
non-detectors above useful intrusion detectors.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


FPR_BUDGETS = (1e-4, 5e-4, 1e-3, 5e-3, 1e-2)


def _as_binary(values: Iterable, attack_label: int = 1) -> np.ndarray:
    arr = np.asarray(values)
    return (arr == attack_label).astype(int)


def compute_binary_metrics(y_true, y_pred, attack_label: int = 1) -> dict[str, float]:
    """Compute attack-centric binary metrics.

    Parameters
    ----------
    y_true, y_pred:
        Labels or predictions. `attack_label` is mapped to positive class 1.
    attack_label:
        Label value representing attack/malicious traffic.
    """
    yt = _as_binary(y_true, attack_label)
    yp = _as_binary(y_pred, attack_label)
    tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
    out = {
        "attack_precision": float(precision_score(yt, yp, pos_label=1, zero_division=0)),
        "attack_recall": float(recall_score(yt, yp, pos_label=1, zero_division=0)),
        "attack_f1": float(f1_score(yt, yp, pos_label=1, zero_division=0)),
        "normal_precision": float(precision_score(yt, yp, pos_label=0, zero_division=0)),
        "normal_recall": float(recall_score(yt, yp, pos_label=0, zero_division=0)),
        "normal_f1": float(f1_score(yt, yp, pos_label=0, zero_division=0)),
        "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(yt, yp, average="weighted", zero_division=0)),
        "weighted_recall": float(recall_score(yt, yp, average="weighted", zero_division=0)),
        "accuracy": float(accuracy_score(yt, yp)),
        "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
        "mcc": float(matthews_corrcoef(yt, yp)) if len(np.unique(yp)) > 1 or len(np.unique(yt)) > 1 else 0.0,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "num_pos": int((yt == 1).sum()),
        "num_neg": int((yt == 0).sum()),
        "positive_rate": float((yt == 1).mean()) if len(yt) else np.nan,
    }
    return out


def _recall_at_fpr(y_true: np.ndarray, y_score: np.ndarray, budget: float) -> tuple[float, float, float, float]:
    order = np.argsort(-y_score)
    ys = y_score[order]
    yt = y_true[order]
    pos_total = max(int((y_true == 1).sum()), 1)
    neg_total = max(int((y_true == 0).sum()), 1)
    unique_end = np.r_[np.where(ys[:-1] != ys[1:])[0], len(ys) - 1]
    tp = np.cumsum(yt == 1)[unique_end].astype(float)
    fp = np.cumsum(yt == 0)[unique_end].astype(float)
    recall = tp / pos_total
    fpr = fp / neg_total
    precision = tp / np.maximum(tp + fp, 1.0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(recall), where=(precision + recall) > 0)
    valid = np.where(fpr <= budget)[0]
    if len(valid) == 0:
        return 0.0, 0.0, 0.0, 1.0
    best = valid[np.lexsort((f1[valid], recall[valid]))[-1]]
    return float(recall[best]), float(precision[best]), float(f1[best]), float(fpr[best])


def compute_score_metrics(y_true, y_score, attack_label: int = 1, budgets: Iterable[float] = FPR_BUDGETS) -> dict[str, float]:
    yt = _as_binary(y_true, attack_label)
    ys = np.asarray(y_score, dtype=float)
    out = {"auroc": np.nan, "aupr": np.nan}
    if len(np.unique(yt)) > 1 and len(np.unique(ys)) > 1:
        out["auroc"] = float(roc_auc_score(yt, ys))
        out["aupr"] = float(average_precision_score(yt, ys))
    for b in budgets:
        suffix = f"{b:.0e}".replace("-", "_")
        r, p, f, actual = _recall_at_fpr(yt, ys, float(b))
        out[f"recall_at_fpr_{suffix}"] = r
        out[f"precision_at_fpr_{suffix}"] = p
        out[f"f1_at_fpr_{suffix}"] = f
        out[f"actual_fpr_{suffix}"] = actual
    return out


def compute_trivial_baselines(y_true, attack_label: int = 1, seed: int = 42) -> pd.DataFrame:
    yt = _as_binary(y_true, attack_label)
    rng = np.random.default_rng(seed)
    base_rate = float(yt.mean()) if len(yt) else 0.0
    specs = {
        "predict_all_normal": np.zeros_like(yt),
        "predict_all_attack": np.ones_like(yt),
        "random_base_rate": (rng.random(len(yt)) < base_rate).astype(int),
        "random_0.1_percent_attack": (rng.random(len(yt)) < 0.001).astype(int),
        "random_1_percent_attack": (rng.random(len(yt)) < 0.01).astype(int),
    }
    rows = []
    for name, pred in specs.items():
        rows.append({"baseline": name, **compute_binary_metrics(yt, pred)})
    return pd.DataFrame(rows)


def compute_ranking_inversion(metrics_df: pd.DataFrame, setting_cols: Iterable[str] = ("dataset", "setting")) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rank models by competing metrics and summarize rank disagreement."""
    df = metrics_df.copy()
    group_cols = [c for c in setting_cols if c in df.columns]
    if not group_cols:
        group_cols = ["dataset"] if "dataset" in df.columns else []
    rank_metrics = ["accuracy", "weighted_f1", "attack_f1", "aupr", "recall_at_fpr_1e_3"]
    ranked = []
    summaries = []
    for key, g in df.groupby(group_cols, dropna=False) if group_cols else [((), df)]:
        gg = g.copy()
        for m in rank_metrics:
            if m in gg.columns:
                gg[f"rank_by_{m}"] = pd.to_numeric(gg[m], errors="coerce").rank(ascending=False, method="min", na_option="bottom")
        if "rank_by_weighted_f1" in gg and "rank_by_attack_f1" in gg:
            gg["rank_gap_weighted_vs_attack"] = gg["rank_by_attack_f1"] - gg["rank_by_weighted_f1"]
            gg["rank_inversion_magnitude"] = gg["rank_gap_weighted_vs_attack"].abs()
        ranked.append(gg)
        prefix = dict(zip(group_cols, key if isinstance(key, tuple) else (key,)))
        summary = dict(prefix)
        x = pd.to_numeric(gg.get("weighted_f1"), errors="coerce")
        y = pd.to_numeric(gg.get("attack_f1"), errors="coerce")
        mask = x.notna() & y.notna()
        if mask.sum() >= 3:
            summary["spearman_weighted_f1_vs_attack_f1"] = float(spearmanr(x[mask], y[mask]).correlation)
            summary["kendall_weighted_f1_vs_attack_f1"] = float(kendalltau(x[mask], y[mask]).correlation)
        else:
            summary["spearman_weighted_f1_vs_attack_f1"] = np.nan
            summary["kendall_weighted_f1_vs_attack_f1"] = np.nan
        for k in (3, 5):
            top_w = set(gg.sort_values("weighted_f1", ascending=False).head(k).get("model", gg.index).astype(str))
            top_a = set(gg.sort_values("attack_f1", ascending=False).head(k).get("model", gg.index).astype(str))
            summary[f"top{k}_overlap_weighted_vs_attack"] = len(top_w & top_a) / max(k, 1)
        summaries.append(summary)
    return pd.concat(ranked, ignore_index=True), pd.DataFrame(summaries)


def compute_event_metrics(prediction_df: pd.DataFrame) -> dict[str, float]:
    """Compute simple event metrics from prediction rows with event_id or labels.

    This function is intentionally conservative. If no event identifiers exist,
    consecutive positive rows are treated as approximate events.
    """
    df = prediction_df.copy()
    if "label" not in df or "prediction" not in df:
        return {"event_recall": np.nan, "event_boundary_source": "missing_label_or_prediction"}
    if "event_id" not in df:
        positive = df["label"].astype(int).eq(1)
        starts = positive & ~positive.shift(fill_value=False)
        df["event_id"] = np.where(positive, starts.cumsum(), np.nan)
        source = "approximate_from_label_transitions"
    else:
        source = "provided_event_id"
    events = df[df["label"].astype(int).eq(1)].groupby("event_id", dropna=True)
    recalled = []
    delays = []
    for _, g in events:
        hit = g[g["prediction"].astype(int).eq(1)]
        recalled.append(len(hit) > 0)
        if len(hit) and "timestamp" in g:
            delays.append(float(hit["timestamp"].iloc[0] - g["timestamp"].iloc[0]))
    fp = int(((df["label"].astype(int) == 0) & (df["prediction"].astype(int) == 1)).sum())
    n = max(len(df), 1)
    out = {
        "event_recall": float(np.mean(recalled)) if recalled else np.nan,
        "mean_detection_delay": float(np.mean(delays)) if delays else np.nan,
        "median_detection_delay": float(np.median(delays)) if delays else np.nan,
        "false_alarm_per_100k": float(fp / n * 100000),
        "event_boundary_source": source,
    }
    if "timestamp" in df and df["timestamp"].notna().sum() > 1:
        duration_h = (float(df["timestamp"].max()) - float(df["timestamp"].min())) / 3600.0
        out["false_alarm_per_hour"] = float(fp / duration_h) if duration_h > 0 else np.nan
    else:
        out["false_alarm_per_hour"] = np.nan
    return out

