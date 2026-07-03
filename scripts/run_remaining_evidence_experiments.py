from __future__ import annotations

import time
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cmf_can.analysis.final_grain_can_revision import (
    FPR_BUDGETS,
    aggregate_train,
    compute_metrics,
    recall_at_budget,
)
from cmf_can.analysis.granularity_shift import collect_window_train, eval_window_model
from cmf_can.analysis.protocol_rescue import FEATURES, TEST_FOLDERS, collect_train, iter_test_folder


OUT = Path("results/final_evidence_completion")
TABLES = OUT / "tables"
FIGS = OUT / "figures"
LOGS = OUT / "logs"

FEATURE_MASKS = {
    "full_safe_can": list(range(len(FEATURES))),
    "without_delta_t_same_id": [i for i, f in enumerate(FEATURES) if f != "delta_t_same_id"],
    "without_payload_delta_l1": [i for i, f in enumerate(FEATURES) if f != "payload_delta_l1"],
    "without_payload_statistics": [i for i, f in enumerate(FEATURES) if f not in {"payload_sum", "payload_mean", "payload_std"}],
    "without_can_id": [i for i, f in enumerate(FEATURES) if f != "can_id"],
    "without_payload_bytes": [i for i, f in enumerate(FEATURES) if not f.startswith("data")],
    "only_timing": [FEATURES.index("delta_t_global"), FEATURES.index("delta_t_same_id")],
    "only_payload": [i for i, f in enumerate(FEATURES) if f.startswith("data") or f in {"payload_sum", "payload_mean", "payload_std", "payload_delta_l1"}],
    "only_id": [FEATURES.index("can_id")],
}


def setup() -> None:
    for path in [OUT, TABLES, FIGS, LOGS]:
        path.mkdir(parents=True, exist_ok=True)


def write_table(df: pd.DataFrame, name: str) -> pd.DataFrame:
    df.to_csv(TABLES / f"{name}.csv", index=False)
    (TABLES / f"{name}.tex").write_text(
        df.to_latex(index=False, escape=True, na_rep="NA", float_format=lambda x: f"{x:.4f}"),
        encoding="utf-8",
    )
    return df


def threshold_for_fpr_budget(y: np.ndarray, score: np.ndarray, budget: float) -> tuple[float, float]:
    order = np.argsort(-score)
    y_sorted = y[order]
    s_sorted = score[order]
    neg = max(int((y == 0).sum()), 1)
    fp = 0
    threshold = np.inf
    actual_fpr = 0.0
    for label, s in zip(y_sorted, s_sorted):
        next_fp = fp + int(label == 0)
        next_fpr = next_fp / neg
        if next_fpr <= budget:
            fp = next_fp
            threshold = float(s)
            actual_fpr = next_fpr
        else:
            break
    return threshold, actual_fpr


def threshold_from_f1(y: np.ndarray, score: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y, score)
    if len(thresholds) == 0:
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.nanargmax(f1))])


def binary_at_threshold(y: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    pred = (score >= threshold).astype(np.int8)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "actual_fpr": fp / max(fp + tn, 1),
        "fnr": fn / max(fn + tp, 1),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def eval_aggregate_model_on_tests(scaler, model, threshold_map: dict[str, float], window_size: int = 100) -> pd.DataFrame:
    rows = []
    for setting, folder in TEST_FOLDERS.items():
        print(f"[remaining] eval aggregate validation-threshold {setting}", flush=True)
        y, score, _ = eval_window_model(model, scaler, 0.5, window_size, setting, folder)
        for name, threshold in threshold_map.items():
            row = {
                "setting": setting,
                "model": "GRAIN_window100_GradientBoosting",
                "granularity": "aggregate_window100",
                "threshold_type": name,
                "threshold": threshold,
                "auroc": roc_auc_score(y, score) if len(np.unique(y)) == 2 else np.nan,
                "aupr": average_precision_score(y, score) if len(np.unique(y)) == 2 else np.nan,
                "num_positive": int((y == 1).sum()),
                "num_negative": int((y == 0).sum()),
            }
            row.update(binary_at_threshold(y, score, threshold))
            rows.append(row)
        for budget in FPR_BUDGETS:
            upper = recall_at_budget(y, score, budget)
            rows.append({
                "setting": setting,
                "model": "GRAIN_window100_GradientBoosting",
                "granularity": "aggregate_window100",
                "threshold_type": "best_test_upper_bound",
                "fpr_budget": budget,
                "threshold": upper["threshold"],
                "precision": upper["precision"],
                "recall": upper["recall"],
                "f1": upper["f1"],
                "actual_fpr": upper["actual_fpr"],
                "auroc": roc_auc_score(y, score) if len(np.unique(y)) == 2 else np.nan,
                "aupr": average_precision_score(y, score) if len(np.unique(y)) == 2 else np.nan,
                "num_positive": int((y == 1).sum()),
                "num_negative": int((y == 0).sum()),
            })
    return pd.DataFrame(rows)


def formal_validation_low_fpr() -> pd.DataFrame:
    print("[remaining] train aggregate window100 for formal validation-threshold low-FPR", flush=True)
    scaler, model, threshold_f1, threshold_low_default, x_val, y_val, val_score = aggregate_train(w=100, seed=42, neg_cap=160_000)
    thresholds = {"validation_best_f1": threshold_f1}
    for budget in FPR_BUDGETS:
        thr, actual = threshold_for_fpr_budget(y_val, val_score, budget)
        thresholds[f"validation_fpr_budget_{budget:g}"] = thr
    out = eval_aggregate_model_on_tests(scaler, model, thresholds, 100)
    out.insert(0, "experiment", "formal_validation_low_fpr")
    write_table(out, "formal_validation_low_fpr")
    (OUT / "formal_validation_low_fpr.md").write_text(
        "# Formal Validation-Threshold Low-FPR\n\n"
        "This experiment selects thresholds on the CT&T train-derived validation split and applies them unchanged to CT&T test01-test04. "
        "Rows labelled `best_test_upper_bound` are retained separately as score-separability upper bounds and are not formal deployment thresholds.\n",
        encoding="utf-8",
    )
    return out


def train_sample_model_from_arrays(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
    feature_cols: list[int] | None = None,
):
    cols = feature_cols if feature_cols is not None else list(range(x_train.shape[1]))
    scaler = StandardScaler().fit(x_train[:, cols])
    model = GradientBoostingClassifier(n_estimators=35, max_depth=2, random_state=seed)
    start = time.time()
    model.fit(scaler.transform(x_train[:, cols]), y_train)
    val_score = model.predict_proba(scaler.transform(x_val[:, cols]))[:, 1]
    threshold = threshold_from_f1(y_val, val_score)
    return scaler, model, threshold, cols, time.time() - start, int(y_train.sum()), int((y_train == 0).sum())


def train_sample_model(seed: int, max_neg_train: int = 120_000, max_neg_val: int = 60_000, feature_cols: list[int] | None = None):
    x_train, y_train, x_val, y_val = collect_train(max_neg_train=max_neg_train, max_neg_val=max_neg_val, seed=seed)
    return train_sample_model_from_arrays(x_train, y_train, x_val, y_val, seed, feature_cols)


def load_sample_test(setting: str) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for _, x, y in iter_test_folder(TEST_FOLDERS[setting]):
        xs.append(x)
        ys.append(y)
    return np.vstack(xs), np.concatenate(ys)


def multiseed_feature_removal() -> pd.DataFrame:
    rows = []
    x_test, y_test = load_sample_test("ctt_test04")
    for seed in [42, 2024, 2026]:
        print(f"[remaining] collect train once for ablation seed={seed}", flush=True)
        x_train, y_train, x_val, y_val = collect_train(max_neg_train=120_000, max_neg_val=60_000, seed=seed)
        for ablation, cols in FEATURE_MASKS.items():
            print(f"[remaining] multiseed ablation seed={seed} ablation={ablation}", flush=True)
            scaler, model, threshold, cols, fit_time, n_pos, n_neg = train_sample_model_from_arrays(x_train, y_train, x_val, y_val, seed, cols)
            score = model.predict_proba(scaler.transform(x_test[:, cols]))[:, 1]
            row = {
                "experiment": "multiseed_feature_removal",
                "setting": "ctt_test04",
                "seed": seed,
                "ablation": ablation,
                "features_used": "|".join(FEATURES[i] for i in cols),
                "num_features": len(cols),
                "threshold_type": "validation_best_f1",
                "threshold": threshold,
                "fit_time_sec": fit_time,
                "num_train_pos": n_pos,
                "num_train_neg": n_neg,
                "auroc": roc_auc_score(y_test, score) if len(np.unique(y_test)) == 2 else np.nan,
                "aupr": average_precision_score(y_test, score) if len(np.unique(y_test)) == 2 else np.nan,
                "recall_at_fpr_1e_3": recall_at_budget(y_test, score, 1e-3)["recall"],
                "num_test_pos": int((y_test == 1).sum()),
                "num_test_neg": int((y_test == 0).sum()),
            }
            row.update(binary_at_threshold(y_test, score, threshold))
            rows.append(row)
    out = pd.DataFrame(rows)
    write_table(out, "grain_multiseed_feature_removal")
    summary = out.groupby("ablation", as_index=False).agg(
        attack_f1_mean=("f1", "mean"),
        attack_f1_std=("f1", "std"),
        aupr_mean=("aupr", "mean"),
        recall_at_fpr_1e_3_mean=("recall_at_fpr_1e_3", "mean"),
    ).sort_values("attack_f1_mean", ascending=False)
    write_table(summary, "grain_multiseed_feature_removal_summary")
    (OUT / "grain_multiseed_feature_removal.md").write_text(
        "# GRAIN Multi-Seed Feature-Removal Retraining\n\n"
        "Each row retrains the sample-level GradientBoosting baseline under a feature mask for seeds 42/2024/2026, using validation-selected F1 thresholds and full CT&T test04 evaluation.\n",
        encoding="utf-8",
    )
    return out


def chunked_negative_ensemble() -> pd.DataFrame:
    print("[remaining] train chunked-negative ensemble", flush=True)
    seeds = [42, 2024, 2026, 7, 99]
    members = []
    for seed in seeds:
        scaler, model, threshold, cols, fit_time, n_pos, n_neg = train_sample_model(seed, 120_000, 60_000, None)
        members.append((seed, scaler, model, threshold, cols, n_pos, n_neg))
    rows = []
    for setting in ["ctt_test02", "ctt_test04"]:
        x_test, y_test = load_sample_test(setting)
        member_scores = []
        for seed, scaler, model, threshold, cols, n_pos, n_neg in members:
            score = model.predict_proba(scaler.transform(x_test[:, cols]))[:, 1]
            member_scores.append(score)
            row = {
                "experiment": "chunked_negative_member",
                "setting": setting,
                "seed": seed,
                "model": "GradientBoosting",
                "negative_chunk_cap": n_neg,
                "threshold_type": "member_validation_best_f1",
                "threshold": threshold,
                "auroc": roc_auc_score(y_test, score) if len(np.unique(y_test)) == 2 else np.nan,
                "aupr": average_precision_score(y_test, score) if len(np.unique(y_test)) == 2 else np.nan,
                "recall_at_fpr_1e_3": recall_at_budget(y_test, score, 1e-3)["recall"],
                "num_test_pos": int((y_test == 1).sum()),
                "num_test_neg": int((y_test == 0).sum()),
            }
            row.update(binary_at_threshold(y_test, score, threshold))
            rows.append(row)
        ens_score = np.mean(np.vstack(member_scores), axis=0)
        # Ensemble threshold is selected from averaged train-validation member scores approximately by reusing 0.5.
        # Budget rows below provide threshold-free operating evidence.
        for budget in FPR_BUDGETS:
            b = recall_at_budget(y_test, ens_score, budget)
            rows.append({
                "experiment": "chunked_negative_ensemble_upper_bound",
                "setting": setting,
                "seed": "42|2024|2026|7|99",
                "model": "GradientBoosting_5chunk_score_average",
                "negative_chunk_cap": sum(m[6] for m in members),
                "threshold_type": "best_test_upper_bound",
                "fpr_budget": budget,
                "threshold": b["threshold"],
                "precision": b["precision"],
                "recall": b["recall"],
                "f1": b["f1"],
                "actual_fpr": b["actual_fpr"],
                "auroc": roc_auc_score(y_test, ens_score) if len(np.unique(y_test)) == 2 else np.nan,
                "aupr": average_precision_score(y_test, ens_score) if len(np.unique(y_test)) == 2 else np.nan,
                "recall_at_fpr_1e_3": recall_at_budget(y_test, ens_score, 1e-3)["recall"],
                "num_test_pos": int((y_test == 1).sum()),
                "num_test_neg": int((y_test == 0).sum()),
            })
    out = pd.DataFrame(rows)
    write_table(out, "chunked_negative_ensemble")
    (OUT / "chunked_negative_ensemble.md").write_text(
        "# Chunked Negative Ensemble\n\n"
        "This experiment trains five GradientBoosting members using different capped negative chunks and averages their scores. "
        "It is a practical chunked-negative stability probe that covers more negative examples than a single capped run; rows labelled upper-bound still use test score budgets and are not validation-tuned deployment thresholds.\n",
        encoding="utf-8",
    )
    return out


def plot_summary() -> None:
    for csv_name, label_col, value_col, fig_name, title in [
        ("formal_validation_low_fpr.csv", "threshold_type", "recall", "formal_validation_low_fpr", "Validation-threshold Recall"),
        ("grain_multiseed_feature_removal_summary.csv", "ablation", "attack_f1_mean", "grain_multiseed_feature_removal", "GRAIN Feature Removal"),
        ("chunked_negative_ensemble.csv", "experiment", "recall_at_fpr_1e_3", "chunked_negative_ensemble", "Chunked Negative Stability"),
    ]:
        path = TABLES / csv_name
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if value_col not in df or label_col not in df:
            continue
        d = df.dropna(subset=[value_col]).copy().head(24)
        fig, ax = plt.subplots(figsize=(8, 3.2))
        ax.bar(np.arange(len(d)), d[value_col].astype(float), color="#4C78A8", edgecolor="black", linewidth=0.7, hatch="//")
        ax.set_xticks(np.arange(len(d)))
        ax.set_xticklabels(d[label_col].astype(str), rotation=35, ha="right", fontsize=7)
        ax.set_ylabel(value_col)
        ax.set_title(title)
        ax.grid(axis="y", color="#E6E6E6", linewidth=0.6)
        fig.tight_layout()
        fig.savefig(FIGS / f"{fig_name}.svg", format="svg")
        plt.close(fig)


def write_final_report() -> None:
    parts = []
    for name in ["formal_validation_low_fpr", "grain_multiseed_feature_removal_summary", "chunked_negative_ensemble"]:
        path = TABLES / f"{name}.csv"
        if path.exists():
            df = pd.read_csv(path)
            parts.append(f"- `{name}` rows: {len(df)}")
    (OUT / "remaining_experiments_completion_report.md").write_text(
        "# Remaining Experiments Completion Report\n\n"
        + "\n".join(parts)
        + "\n\nThese experiments strengthen the paper evidence chain by adding validation-threshold low-FPR rows, multi-seed feature-removal retraining, and a chunked-negative ensemble stability probe.\n",
        encoding="utf-8",
    )


def main() -> None:
    setup()
    if not (TABLES / "formal_validation_low_fpr.csv").exists():
        formal_validation_low_fpr()
    else:
        print("[remaining] skip existing formal_validation_low_fpr.csv", flush=True)
    multiseed_feature_removal()
    chunked_negative_ensemble()
    plot_summary()
    write_final_report()


if __name__ == "__main__":
    main()
