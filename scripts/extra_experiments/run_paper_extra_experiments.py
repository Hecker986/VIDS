from __future__ import annotations

import argparse
import json
import os
import pickle
import platform
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-vids")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from cmf_can.analysis.protocol_rescue import FEATURES, RAW, TEST_FOLDERS, read_ctt_file


ROOT = Path("results/paper_rewrite_framework_style/extra_experiments")
TABLES = ROOT / "tables"
FIGS = ROOT / "figures"
REPORTS = ROOT / "reports"
PREDS = ROOT / "predictions"
CONFIGS = ROOT / "configs"
LOGS = ROOT / "logs"

FPR_BUDGETS = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
FPR_SUFFIX = {
    1e-4: "1e_4",
    5e-4: "5e_4",
    1e-3: "1e_3",
    5e-3: "5e_3",
    1e-2: "1e_2",
}
SETTING_ORDER = ["ctt_test01", "ctt_test02", "ctt_test03", "ctt_test04"]
SETTING_LABELS = {
    "ctt_test01": "Test01",
    "ctt_test02": "Test02",
    "ctt_test03": "Test03",
    "ctt_test04": "Test04",
}
SHIFT_META = {
    "ctt_test01": ("known", "known"),
    "ctt_test02": ("unknown", "known"),
    "ctt_test03": ("known", "unknown"),
    "ctt_test04": ("unknown", "unknown"),
}
PALETTE = {
    "Raw-window+GB": "#6F86A5",
    "GRAIN+GB": "#2F8C7D",
    "GB-sample": "#B98A2E",
    "Raw-Trans": "#B96B6C",
    "GRAIN-W100": "#2F8C7D",
    "Full GRAIN": "#2F8C7D",
    "Timing only": "#2F5E8E",
    "HistGradientBoosting": "#876BA3",
    "RandomForest": "#7A8A49",
    "LogisticRegression": "#A96C3C",
    "MLP": "#A95C7B",
}


def setup_style() -> None:
    for p in [ROOT, TABLES, FIGS, REPORTS, PREDS, CONFIGS, LOGS]:
        p.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans", "sans-serif"],
            "font.size": 8.5,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "legend.frameon": False,
        }
    )


def save_svg(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIGS / f"{name}.svg", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def write_table(df: pd.DataFrame, name: str) -> None:
    df.to_csv(TABLES / f"{name}.csv", index=False)
    tex = df.to_latex(index=False, escape=True, na_rep="NA", float_format=lambda x: f"{x:.4f}")
    (TABLES / f"{name}.tex").write_text(tex, encoding="utf-8")


def threshold_from_val(y: np.ndarray, score: np.ndarray) -> float:
    p, r, thresholds = precision_recall_curve(y, score)
    if len(thresholds) == 0:
        return 0.5
    f1 = 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-12)
    return float(thresholds[int(np.nanargmax(f1))])


def recall_at_fpr(y: np.ndarray, score: np.ndarray, budget: float) -> dict[str, float]:
    order = np.argsort(-score)
    ys = y[order]
    ss = score[order]
    pos = max(int((y == 1).sum()), 1)
    neg = max(int((y == 0).sum()), 1)
    tp = 0
    fp = 0
    best = {"recall": 0.0, "precision": 1.0, "f1": 0.0, "actual_fpr": 0.0, "threshold": float("inf")}
    for label, threshold in zip(ys, ss):
        if int(label) == 1:
            tp += 1
        else:
            fp += 1
        fpr = fp / neg
        if fpr <= budget:
            rec = tp / pos
            prec = tp / max(tp + fp, 1)
            best = {
                "recall": float(rec),
                "precision": float(prec),
                "f1": float(2 * prec * rec / max(prec + rec, 1e-12)),
                "actual_fpr": float(fpr),
                "threshold": float(threshold),
            }
        else:
            break
    return best


def metrics_from_score(y: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, float | int]:
    pred = (score >= threshold).astype(np.int8)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    out: dict[str, float | int] = {
        "accuracy": (tp + tn) / max(tp + fp + tn + fn, 1),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fnr": fn / max(fn + tp, 1),
        "fpr": fp / max(fp + tn, 1),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "aupr": average_precision_score(y, score) if len(np.unique(y)) == 2 else np.nan,
        "auroc": roc_auc_score(y, score) if len(np.unique(y)) == 2 else np.nan,
    }
    for budget in FPR_BUDGETS:
        key = FPR_SUFFIX[budget]
        out[f"recall_at_fpr_{key}"] = recall_at_fpr(y, score, budget)["recall"]
    return out


def feature_indices(names: list[str]) -> list[int]:
    return [FEATURES.index(n) for n in names]


RAW_COLS = feature_indices(["can_id", "dlc", *[f"data{i}" for i in range(8)], "delta_t_global"])
GRAIN_COLS = list(range(len(FEATURES)))
GRAIN_RESIDUAL_COLS = feature_indices(["delta_t_global", "delta_t_same_id", "payload_sum", "payload_mean", "payload_std", "payload_delta_l1"])


def window_matrix(x: np.ndarray, y: np.ndarray, w: int, cols: list[int], include_id_entropy: bool = True) -> tuple[np.ndarray, np.ndarray]:
    n = len(y) // w
    if n == 0:
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int8)
    xx = x[: n * w][:, cols].reshape(n, w, len(cols))
    yy = y[: n * w].reshape(n, w)
    last = xx[:, -1, :]
    mean = xx.mean(axis=1)
    std = xx.std(axis=1)
    mx = xx.max(axis=1)
    mn = xx.min(axis=1)
    feats = [last, mean, std, mx, mn]
    if include_id_entropy and FEATURES.index("can_id") in cols:
        local_id_pos = cols.index(FEATURES.index("can_id"))
        ent = np.zeros(n, dtype=np.float32)
        top = np.zeros(n, dtype=np.float32)
        ids = xx[:, :, local_id_pos]
        for i in range(n):
            _, counts = np.unique(ids[i], return_counts=True)
            p = counts.astype(np.float32) / max(counts.sum(), 1)
            ent[i] = float(-(p * np.log2(np.maximum(p, 1e-12))).sum())
            top[i] = float(p.max())
        feats.append(ent[:, None])
        feats.append(top[:, None])
    label = (yy.max(axis=1) > 0).astype(np.int8)
    return np.concatenate(feats, axis=1).astype(np.float32), label


def aggregate_column_indices(base_cols: list[int], full_cols: list[int] = GRAIN_COLS) -> list[int]:
    """Map frame-level feature indices to columns in window_matrix(full_cols)."""
    rel = [full_cols.index(c) for c in base_cols if c in full_cols]
    width = len(full_cols)
    out: list[int] = []
    for block in range(5):
        out.extend([block * width + r for r in rel])
    if FEATURES.index("can_id") in base_cols:
        out.extend([5 * width, 5 * width + 1])
    return out


def load_window_split(
    cols: list[int],
    w: int = 100,
    seed: int = 42,
    max_neg_train: int = 120_000,
    max_neg_val: int = 60_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x_train: list[np.ndarray] = []
    y_train: list[np.ndarray] = []
    x_val: list[np.ndarray] = []
    y_val: list[np.ndarray] = []
    for path in sorted((RAW / "train_01").glob("*.csv")):
        x, y, _ = read_ctt_file(path)
        wx, wy = window_matrix(x, y, w, cols)
        if len(wy) == 0:
            continue
        pos = np.where(wy == 1)[0]
        neg = np.where(wy == 0)[0]
        is_val = path.stem.endswith("-2")
        cap = max_neg_val if is_val else max_neg_train
        per_file_cap = max(1, cap // 6)
        if len(neg) > per_file_cap:
            neg = rng.choice(neg, size=per_file_cap, replace=False)
        idx = np.concatenate([pos, neg])
        rng.shuffle(idx)
        if is_val:
            x_val.append(wx[idx])
            y_val.append(wy[idx])
        else:
            x_train.append(wx[idx])
            y_train.append(wy[idx])
    return np.vstack(x_train), np.concatenate(y_train), np.vstack(x_val), np.concatenate(y_val)


def iter_setting_windows(setting: str, cols: list[int], w: int = 100):
    for path in sorted((RAW / TEST_FOLDERS[setting]).glob("*.csv")):
        x, y, _ = read_ctt_file(path)
        wx, wy = window_matrix(x, y, w, cols)
        if len(wy):
            yield path, wx, wy


def model_score(model, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    if hasattr(model, "decision_function"):
        s = model.decision_function(x)
        return 1.0 / (1.0 + np.exp(-s))
    return model.predict(x).astype(float)


def train_classifier(name: str, seed: int = 42):
    if name == "GradientBoosting":
        return GradientBoostingClassifier(n_estimators=35, max_depth=2, random_state=seed)
    if name == "HistGradientBoosting":
        return HistGradientBoostingClassifier(max_iter=60, max_leaf_nodes=31, random_state=seed)
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=60, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=-1, random_state=seed)
    if name == "LogisticRegression":
        return LogisticRegression(max_iter=300, class_weight="balanced")
    if name == "MLP":
        return MLPClassifier(hidden_layer_sizes=(48,), max_iter=25, early_stopping=True, random_state=seed)
    raise ValueError(f"unknown classifier: {name}")


def fit_window_pipeline(cols: list[int], classifier: str = "GradientBoosting", w: int = 100, seed: int = 42):
    x_train, y_train, x_val, y_val = load_window_split(cols, w=w, seed=seed)
    scaler = StandardScaler().fit(x_train)
    model = train_classifier(classifier, seed=seed)
    t0 = time.perf_counter()
    model.fit(scaler.transform(x_train), y_train)
    fit_sec = time.perf_counter() - t0
    val_score = model_score(model, scaler.transform(x_val))
    threshold = threshold_from_val(y_val, val_score)
    return scaler, model, threshold, fit_sec, len(y_train), int(y_train.sum()), len(y_val), int(y_val.sum())


def eval_window_pipeline(
    setting: str,
    cols: list[int],
    scaler,
    model,
    threshold: float,
    pipeline_label: str,
    representation: str,
    classifier: str,
    w: int = 100,
    source_script: str = "scripts/extra_experiments/run_paper_extra_experiments.py",
    save_prediction: bool = True,
) -> dict[str, object]:
    ys: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    files: list[str] = []
    for path, wx, wy in iter_setting_windows(setting, cols, w=w):
        score = model_score(model, scaler.transform(wx))
        ys.append(wy)
        scores.append(score)
        if save_prediction:
            files.extend([str(path)] * len(wy))
    y = np.concatenate(ys)
    score = np.concatenate(scores)
    m = metrics_from_score(y, score, threshold)
    pred_path = ""
    if save_prediction:
        pred_path = str(PREDS / f"{setting}_{pipeline_label.replace('+', '_').replace(' ', '_')}_scores.csv")
        pd.DataFrame(
            {
                "sample_id": np.arange(len(y)),
                "setting": setting,
                "pipeline": pipeline_label,
                "representation": representation,
                "classifier": classifier,
                "label": y,
                "score": score,
                "prediction": (score >= threshold).astype(int),
                "threshold": threshold,
                "source_file": files,
            }
        ).to_csv(pred_path, index=False)
    return {
        "setting": setting,
        "pipeline": pipeline_label,
        "representation": representation,
        "window_size": w,
        "classifier": classifier,
        "threshold_rule": "validation_f1",
        "score_available": True,
        **m,
        "source_script": source_script,
        "source_prediction_file": pred_path,
        "note": "same train/test split, same classifier family, attack-positive metrics",
    }


def run_raw_same_classifier_control() -> pd.DataFrame:
    rows = []
    configs = [
        ("Raw-window+GB", "raw CAN window statistics without same-ID residual features", RAW_COLS),
        ("GRAIN+GB", "same-ID residual and payload-statistic features before window aggregation", GRAIN_RESIDUAL_COLS),
    ]
    for pipeline, representation, cols in configs:
        print(f"[extra] fit {pipeline}", flush=True)
        scaler, model, threshold, *_ = fit_window_pipeline(cols, classifier="GradientBoosting")
        for setting in SETTING_ORDER:
            print(f"[extra] eval {pipeline} {setting}", flush=True)
            rows.append(eval_window_pipeline(setting, cols, scaler, model, threshold, pipeline, representation, "GradientBoosting"))
    out = pd.DataFrame(rows)
    write_table(out, "raw_same_classifier_control")
    plot_grouped_bar(out, "raw_same_classifier_control", "pipeline", "f1", "Attack F1")
    REPORTS.joinpath("raw_same_classifier_control_report.md").write_text(
        "# Raw-window + Same Classifier Control\n\n"
        "This experiment isolates representation from classifier choice. Both rows use GradientBoosting, W100, validation-F1 thresholding, and the same CT&T train/test split. "
        "Raw-window features exclude same-ID residuals; GRAIN features include the causal same-ID timing and payload residual fields.\n",
        encoding="utf-8",
    )
    return out


def run_efficiency_cost(control: pd.DataFrame) -> pd.DataFrame:
    rows = []
    cpu = platform.processor() or platform.machine() or "unknown"
    gpu = "not_used_cpu_only"
    setting = "ctt_test04"
    configs = [
        ("Raw-window+GB", "GradientBoosting", RAW_COLS),
        ("GRAIN+GB", "GradientBoosting", GRAIN_COLS),
    ]
    for pipeline, classifier, cols in configs:
        print(f"[extra] efficiency fit {pipeline}", flush=True)
        scaler, model, threshold, *_ = fit_window_pipeline(cols, classifier=classifier)
        model_size_mb = len(pickle.dumps({"scaler": scaler, "model": model, "threshold": threshold})) / (1024 * 1024)
        feat_times, agg_times, clf_times, e2e_times = [], [], [], []
        frames_total = 0
        windows_total = 0
        for repeat in range(3):
            print(f"[extra] efficiency {pipeline} repeat={repeat+1}/3", flush=True)
            t0 = time.perf_counter()
            wx_parts = []
            wy_parts = []
            feat_ms = 0.0
            agg_ms = 0.0
            n_frames = 0
            measured_files = sorted((RAW / TEST_FOLDERS[setting]).glob("*.csv"))[:2]
            for path in measured_files:
                ft0 = time.perf_counter()
                x, y, _ = read_ctt_file(path)
                feat_ms += (time.perf_counter() - ft0) * 1000
                n_frames += len(y)
                at0 = time.perf_counter()
                wx, wy = window_matrix(x, y, 100, cols)
                agg_ms += (time.perf_counter() - at0) * 1000
                wx_parts.append(wx)
                wy_parts.append(wy)
            wx_all = np.vstack(wx_parts)
            windows = len(wx_all)
            ct0 = time.perf_counter()
            _ = model_score(model, scaler.transform(wx_all))
            clf_ms = (time.perf_counter() - ct0) * 1000
            e2e_ms = (time.perf_counter() - t0) * 1000
            frames_total = n_frames
            windows_total = windows
            feat_times.append(feat_ms)
            agg_times.append(agg_ms)
            clf_times.append(clf_ms)
            e2e_times.append(e2e_ms)
        end_ms = float(np.mean(e2e_times))
        rows.append(
            {
                "pipeline": pipeline,
                "setting": setting,
                "window_size": 100,
                "classifier": classifier,
                "device": "cpu",
                "cpu": cpu,
                "gpu": gpu,
                "num_frames": frames_total,
                "num_windows": windows_total,
                "feature_extraction_ms_mean": float(np.mean(feat_times)),
                "feature_extraction_ms_std": float(np.std(feat_times, ddof=1)),
                "aggregation_ms_mean": float(np.mean(agg_times)),
                "aggregation_ms_std": float(np.std(agg_times, ddof=1)),
                "classifier_inference_ms_mean": float(np.mean(clf_times)),
                "classifier_inference_ms_std": float(np.std(clf_times, ddof=1)),
                "end_to_end_ms_mean": end_ms,
                "end_to_end_ms_std": float(np.std(e2e_times, ddof=1)),
                "frames_per_second": frames_total / max(end_ms / 1000, 1e-12),
                "windows_per_second": windows_total / max(end_ms / 1000, 1e-12),
                "model_size_mb": model_size_mb,
                "peak_memory_mb": np.nan,
                "repeated_runs": 3,
                "source_script": "scripts/extra_experiments/run_paper_extra_experiments.py",
                "note": f"CPU-only inference measurement on first {len(measured_files)} CT&T test04 files; feature extraction includes CSV parsing and causal residual construction.",
            }
        )
    out = pd.DataFrame(rows)
    write_table(out, "efficiency_cost")
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    order = out.sort_values("frames_per_second", ascending=True)
    ax.barh(order["pipeline"], order["frames_per_second"], color=[PALETTE.get(x, "#777") for x in order["pipeline"]], edgecolor="#28313B", linewidth=0.6)
    ax.set_xlabel("End-to-end frames/s (CPU)")
    ax.grid(axis="x", color="#E8EBEF", linewidth=0.6)
    for i, v in enumerate(order["frames_per_second"]):
        ax.text(v * 1.01, i, f"{v:,.0f}", va="center", fontsize=7.6)
    save_svg(fig, "efficiency_cost")
    REPORTS.joinpath("efficiency_cost_report.md").write_text(
        "# Efficiency / Deployment Cost\n\n"
        "CPU-only throughput was measured on CT&T Test04 for three repeated runs. Timing separates feature extraction, window aggregation, and classifier inference. "
        "Feature extraction includes CSV parsing because this is the current reproducible repository path.\n",
        encoding="utf-8",
    )
    return out


def run_cross_shift_matrix(control: pd.DataFrame) -> pd.DataFrame:
    conventional = pd.read_csv("results/paper_revision_final_grain_can/tables/conventional_ids_metrics_ctt.csv")
    rows = []
    for _, r in conventional.iterrows():
        label = None
        model = str(r["model"])
        if model == "GradientBoosting / sample" and str(r.get("source")) == "final_grain_can_a1_negative_stability":
            label = "GB-sample"
        elif model == "transformer / old_window100_deep":
            label = "Raw-Trans"
        elif model == "GradientBoosting / window_100":
            label = "GRAIN-W100"
        if label is None:
            continue
        setting = str(r["setting"])
        vehicle, attack = SHIFT_META[setting]
        rows.append(
            {
                "setting": setting,
                "vehicle_shift": vehicle,
                "attack_shift": attack,
                "model": label,
                "precision": r.get("precision"),
                "recall": r.get("recall"),
                "f1": r.get("f1"),
                "fnr": r.get("fnr"),
                "fpr": r.get("fpr"),
                "aupr": r.get("aupr"),
                "auroc": r.get("auroc"),
                "tp": r.get("tp"),
                "fp": r.get("fp"),
                "tn": r.get("tn"),
                "fn": r.get("fn"),
                "note": "2x2 shift decomposition; Test01-Test04 are categorical settings, not a sequence",
            }
        )
    out = pd.DataFrame(rows).drop_duplicates(["setting", "model"], keep="last")
    write_table(out, "cross_shift_matrix")
    # Heatmap: delta GRAIN minus Raw-Trans
    pivot = out.pivot_table(index="setting", columns="model", values="f1", aggfunc="first")
    delta_rows = []
    for setting in SETTING_ORDER:
        grain = pivot.loc[setting, "GRAIN-W100"] if setting in pivot.index and "GRAIN-W100" in pivot else np.nan
        raw = pivot.loc[setting, "Raw-Trans"] if setting in pivot.index and "Raw-Trans" in pivot else np.nan
        delta_rows.append({"setting": setting, "delta_f1_grain_minus_raw_trans": grain - raw})
    delta = pd.DataFrame(delta_rows)
    for metric in ["recall", "fnr", "fpr"]:
        p = out.pivot_table(index="setting", columns="model", values=metric, aggfunc="first")
        delta[f"delta_{metric}_grain_minus_raw_trans"] = [p.loc[s, "GRAIN-W100"] - p.loc[s, "Raw-Trans"] for s in SETTING_ORDER]
    delta.to_csv(TABLES / "cross_shift_deltas.csv", index=False)
    out = out.merge(delta, on="setting", how="left")
    write_table(out, "cross_shift_matrix")
    plot_cross_shift(out, delta)
    REPORTS.joinpath("cross_shift_matrix_report.md").write_text(
        "# CT&T 2x2 Cross-Shift Matrix\n\n"
        "Test01-Test04 are a 2x2 decomposition of vehicle shift and attack shift, not a temporal series. The SVG outputs use a matrix heatmap and grouped bars rather than line charts.\n",
        encoding="utf-8",
    )
    return out


def plot_cross_shift(out: pd.DataFrame, delta: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 2.8))
    models = ["GB-sample", "Raw-Trans", "GRAIN-W100"]
    x = np.arange(len(SETTING_ORDER))
    width = 0.23
    for i, model in enumerate(models):
        vals = [float(out[(out["setting"].eq(s)) & (out["model"].eq(model))]["f1"].iloc[0]) for s in SETTING_ORDER]
        ax.bar(x + (i - 1) * width, vals, width=width * 0.92, color=PALETTE.get(model, "#777"), edgecolor="#28313B", linewidth=0.55, label=model)
    ax.set_xticks(x)
    ax.set_xticklabels([SETTING_LABELS[s] for s in SETTING_ORDER])
    ax.set_ylabel("Attack F1")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", color="#E8EBEF", linewidth=0.6)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    save_svg(fig, "cross_shift_matrix_f1")
    mat = np.array(
        [
            [
                delta.loc[delta["setting"].eq("ctt_test01"), "delta_f1_grain_minus_raw_trans"].iloc[0],
                delta.loc[delta["setting"].eq("ctt_test03"), "delta_f1_grain_minus_raw_trans"].iloc[0],
            ],
            [
                delta.loc[delta["setting"].eq("ctt_test02"), "delta_f1_grain_minus_raw_trans"].iloc[0],
                delta.loc[delta["setting"].eq("ctt_test04"), "delta_f1_grain_minus_raw_trans"].iloc[0],
            ],
        ]
    )
    fig, ax = plt.subplots(figsize=(3.8, 3.0))
    im = ax.imshow(mat, cmap="BrBG", vmin=-1, vmax=1)
    ax.set_xticks([0, 1], labels=["Known attack", "Unknown attack"])
    ax.set_yticks([0, 1], labels=["Known vehicle", "Unknown vehicle"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{mat[i, j]:+.3f}", ha="center", va="center", fontsize=9, color="#1F2933")
    cbar = fig.colorbar(im, ax=ax, shrink=0.78)
    cbar.set_label("Delta F1: GRAIN-W100 - Raw-Trans")
    save_svg(fig, "cross_shift_delta_f1_heatmap")


def run_low_fpr_operating_curve(control: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for path in sorted(PREDS.glob("*_scores.csv")):
        df = pd.read_csv(path)
        y = df["label"].astype(int).to_numpy()
        score = df["score"].astype(float).to_numpy()
        base = {
            "setting": df["setting"].iloc[0],
            "model": df["pipeline"].iloc[0],
            "score_available": True,
            "aupr": average_precision_score(y, score) if len(np.unique(y)) == 2 else np.nan,
            "auroc": roc_auc_score(y, score) if len(np.unique(y)) == 2 else np.nan,
            "score_file": str(path),
        }
        for budget in FPR_BUDGETS:
            b = recall_at_fpr(y, score, budget)
            rows.append({**base, "fpr_budget": budget, "recall_at_budget": b["recall"], "actual_fpr": b["actual_fpr"], "threshold": b["threshold"], "note": "diagnostic operating curve from model scores"})
    out = pd.DataFrame(rows)
    write_table(out, "low_fpr_operating_curve")
    fig, ax = plt.subplots(figsize=(6.7, 3.0))
    plot = out[out["setting"].isin(["ctt_test03", "ctt_test04"])]
    labels = []
    for (setting, model), g in plot.groupby(["setting", "model"]):
        g = g.sort_values("fpr_budget")
        label = f"{SETTING_LABELS[setting]} {model}"
        labels.append(label)
        ax.plot(g["fpr_budget"], g["recall_at_budget"], marker="o", linewidth=1.2, markersize=3.5, label=label, color=PALETTE.get(model, None))
    ax.set_xscale("log")
    ax.set_xlabel("FPR budget")
    ax.set_ylabel("Recall at budget")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", color="#E8EBEF", linewidth=0.6)
    ax.legend(ncol=2, fontsize=7)
    save_svg(fig, "low_fpr_operating_curve")
    REPORTS.joinpath("low_fpr_operating_curve_report.md").write_text(
        "# Low-FPR Operating Curve\n\n"
        "Rows are computed only for score-producing pipelines from the raw-same-classifier control. These are diagnostic operating curves from scores, not prior-paper comparisons.\n",
        encoding="utf-8",
    )
    return out


VARIANTS = {
    "Full GRAIN": GRAIN_COLS,
    "Timing only": feature_indices(["delta_t_global", "delta_t_same_id"]),
    "w/o same-ID timing": [i for i, f in enumerate(FEATURES) if f != "delta_t_same_id"],
    "w/o payload delta": [i for i, f in enumerate(FEATURES) if f != "payload_delta_l1"],
    "w/o payload stats": [i for i, f in enumerate(FEATURES) if f not in {"payload_sum", "payload_mean", "payload_std"}],
    "w/o CAN-ID behavior": [i for i, f in enumerate(FEATURES) if f != "can_id"],
    "Payload only": feature_indices([*[f"data{i}" for i in range(8)], "payload_sum", "payload_mean", "payload_std", "payload_delta_l1"]),
    "ID only": feature_indices(["can_id"]),
}


def run_feature_ablation_extended() -> pd.DataFrame:
    rows = []
    print("[extra] build cached full GRAIN train/val matrices for ablation", flush=True)
    x_train_full, y_train, x_val_full, y_val = load_window_split(GRAIN_COLS, w=100, seed=42)
    test_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for setting in ["ctt_test02", "ctt_test03", "ctt_test04"]:
        xs, ys = [], []
        for _, wx, wy in iter_setting_windows(setting, GRAIN_COLS, w=100):
            xs.append(wx)
            ys.append(wy)
        test_cache[setting] = (np.vstack(xs), np.concatenate(ys))
    for variant, cols in VARIANTS.items():
        print(f"[extra] ablation fit {variant}", flush=True)
        idx = aggregate_column_indices(cols)
        x_train = x_train_full[:, idx]
        x_val = x_val_full[:, idx]
        scaler = StandardScaler().fit(x_train)
        model = train_classifier("GradientBoosting", seed=42)
        model.fit(scaler.transform(x_train), y_train)
        val_score = model_score(model, scaler.transform(x_val))
        threshold = threshold_from_val(y_val, val_score)
        for setting in ["ctt_test02", "ctt_test03", "ctt_test04"]:
            tx_full, ty = test_cache[setting]
            score = model_score(model, scaler.transform(tx_full[:, idx]))
            m = metrics_from_score(ty, score, threshold)
            rows.append(
                {
                    "setting": setting,
                    "variant": variant,
                    "feature_groups_used": "|".join(FEATURES[i] for i in cols),
                    "classifier": "GradientBoosting",
                    "window_size": 100,
                    **{k: m[k] for k in ["accuracy", "precision", "recall", "f1", "fnr", "fpr", "tp", "fp", "tn", "fn", "aupr", "auroc"]},
                    "recall_at_fpr_1e_3": m.get("recall_at_fpr_1e_3", np.nan),
                    "source_script": "scripts/extra_experiments/run_paper_extra_experiments.py",
                    "note": "full retraining per feature mask from cached full W100 GRAIN matrix; validation-F1 threshold",
                }
            )
    out = pd.DataFrame(rows)
    write_table(out, "feature_ablation_extended")
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.4), sharex=True)
    for ax, setting in zip(axes, ["ctt_test02", "ctt_test03", "ctt_test04"]):
        d = out[out["setting"].eq(setting)].sort_values("f1", ascending=True)
        ax.barh(d["variant"], d["f1"], color=[PALETTE.get(v, "#BFDCD6") for v in d["variant"]], edgecolor="#28313B", linewidth=0.45)
        ax.set_title(SETTING_LABELS[setting], fontsize=9, fontweight="bold")
        ax.set_xlabel("Attack F1")
        ax.grid(axis="x", color="#E8EBEF", linewidth=0.55)
        if ax is not axes[0]:
            ax.set_ylabel("")
            ax.set_yticklabels([])
    save_svg(fig, "feature_ablation_extended")
    REPORTS.joinpath("feature_ablation_extended_report.md").write_text(
        "# Feature Ablation Extension\n\n"
        "Each feature mask is retrained with GradientBoosting on W100 GRAIN-style aggregate features. Results cover Test02, Test03, and Test04 with attack-positive metrics.\n",
        encoding="utf-8",
    )
    return out


def run_classifier_sensitivity() -> pd.DataFrame:
    rows = []
    print("[extra] build cached full GRAIN train/val matrices for classifier sensitivity", flush=True)
    x_train, y_train, x_val, y_val = load_window_split(GRAIN_COLS, w=100, seed=42)
    test_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for setting in ["ctt_test03", "ctt_test04"]:
        xs, ys = [], []
        for _, wx, wy in iter_setting_windows(setting, GRAIN_COLS, w=100):
            xs.append(wx)
            ys.append(wy)
        test_cache[setting] = (np.vstack(xs), np.concatenate(ys))
    classifiers = ["GradientBoosting", "HistGradientBoosting", "RandomForest", "LogisticRegression", "MLP"]
    for clf in classifiers:
        print(f"[extra] classifier sensitivity fit {clf}", flush=True)
        try:
            scaler = StandardScaler().fit(x_train)
            model = train_classifier(clf, seed=42)
            model.fit(scaler.transform(x_train), y_train)
            val_score = model_score(model, scaler.transform(x_val))
            threshold = threshold_from_val(y_val, val_score)
            status = "completed"
            error = ""
        except Exception as exc:
            status = "failed"
            error = str(exc)
            for setting in ["ctt_test03", "ctt_test04"]:
                rows.append({"setting": setting, "representation": "GRAIN-W100", "window_size": 100, "classifier": clf, "hyperparameters": "default_extra_experiment", "status": status, "note": error})
            continue
        for setting in ["ctt_test03", "ctt_test04"]:
            tx, ty = test_cache[setting]
            score = model_score(model, scaler.transform(tx))
            m = metrics_from_score(ty, score, threshold)
            rows.append(
                {
                    "setting": setting,
                    "representation": "GRAIN-W100",
                    "window_size": 100,
                    "classifier": clf,
                    "hyperparameters": json.dumps(model.get_params(), default=str),
                    **{k: m[k] for k in ["accuracy", "precision", "recall", "f1", "fnr", "fpr", "tp", "fp", "tn", "fn", "aupr", "auroc"]},
                    "source_script": "scripts/extra_experiments/run_paper_extra_experiments.py",
                    "status": status,
                    "note": "same GRAIN-W100 representation; validation-F1 threshold",
                }
            )
    out = pd.DataFrame(rows)
    write_table(out, "grain_classifier_sensitivity")
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    d = out[out["setting"].eq("ctt_test04")].dropna(subset=["f1"]).sort_values("f1")
    ax.barh(d["classifier"], d["f1"], color=[PALETTE.get(c, "#BFDCD6") for c in d["classifier"]], edgecolor="#28313B", linewidth=0.5)
    ax.set_xlabel("Attack F1 on Test04")
    ax.grid(axis="x", color="#E8EBEF", linewidth=0.6)
    save_svg(fig, "grain_classifier_sensitivity")
    REPORTS.joinpath("grain_classifier_sensitivity_report.md").write_text(
        "# Classifier Sensitivity\n\n"
        "This experiment fixes the GRAIN-W100 representation and varies only the classifier family. It tests whether the method depends entirely on GradientBoosting.\n",
        encoding="utf-8",
    )
    return out


def plot_grouped_bar(df: pd.DataFrame, name: str, hue_col: str, value_col: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 3.0))
    hues = list(df[hue_col].drop_duplicates())
    x = np.arange(len(SETTING_ORDER))
    width = min(0.72 / max(len(hues), 1), 0.34)
    for i, hue in enumerate(hues):
        vals = []
        for setting in SETTING_ORDER:
            d = df[(df["setting"].eq(setting)) & (df[hue_col].eq(hue))]
            vals.append(float(d[value_col].iloc[0]) if len(d) else np.nan)
        ax.bar(x + (i - (len(hues) - 1) / 2) * width, vals, width=width * 0.92, label=str(hue), color=PALETTE.get(str(hue), "#9AA7B4"), edgecolor="#28313B", linewidth=0.55)
    ax.set_xticks(x)
    ax.set_xticklabels([SETTING_LABELS[s] for s in SETTING_ORDER])
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", color="#E8EBEF", linewidth=0.6)
    ax.legend(ncol=min(3, len(hues)), loc="upper center", bbox_to_anchor=(0.5, 1.16), fontsize=7.5)
    save_svg(fig, name)


def write_report(outputs: dict[str, pd.DataFrame]) -> None:
    lines = [
        "# Extra Experiment Report",
        "",
        "All experiments use CT&T set_01, attack as the positive class, fixed W100 unless otherwise stated, and validation-F1 threshold selection. No test labels are used to select features, windows, classifiers, or thresholds.",
        "",
        "## Completed experiments",
    ]
    for name, df in outputs.items():
        lines.append(f"- {name}: {len(df)} rows")
    raw = outputs.get("raw_same_classifier_control")
    if raw is not None and not raw.empty:
        lines += ["", "## Key numeric checks"]
        for setting in SETTING_ORDER:
            d = raw[raw["setting"].eq(setting)]
            if {"Raw-window+GB", "GRAIN+GB"}.issubset(set(d["pipeline"])):
                r = float(d[d["pipeline"].eq("Raw-window+GB")]["f1"].iloc[0])
                g = float(d[d["pipeline"].eq("GRAIN+GB")]["f1"].iloc[0])
                lines.append(f"- {SETTING_LABELS[setting]} same-classifier F1: Raw-window+GB={r:.4f}, GRAIN+GB={g:.4f}, delta={g-r:+.4f}.")
    clf = outputs.get("grain_classifier_sensitivity")
    if clf is not None and not clf.empty and "ctt_test04" in set(clf.get("setting", [])):
        d = clf[clf["setting"].eq("ctt_test04")].dropna(subset=["f1"]).sort_values("f1", ascending=False)
        if len(d):
            b = d.iloc[0]
            lines.append(f"- Test04 classifier sensitivity best row: {b['classifier']} on fixed GRAIN-W100, F1={float(b['f1']):.4f}, AUPR={float(b['aupr']):.4f}.")
    lines += [
        "",
        "## Evidence impact",
        "- Raw-window+same-classifier control isolates representation from classifier choice.",
        "- Efficiency measurements support or qualify the lightweight deployment claim using CPU-only timing.",
        "- Cross-shift matrix treats Test01-Test04 as vehicle-shift by attack-shift categories, not a sequence.",
        "- Low-FPR curves are diagnostic score operating curves for score-producing models only.",
        "- Feature ablation extension retrains each mask; it should be read as setting-specific evidence.",
        "- Classifier sensitivity fixes GRAIN-W100 features and varies classifier family.",
        "- The classifier sensitivity result should not silently replace the main protocol; it is a new candidate backed by this supplement and should be introduced explicitly if used.",
        "",
        "## Main-text suitability",
        "- P0 raw/same-classifier and cross-shift matrix can go in the main paper or appendix depending on page budget.",
        "- Efficiency belongs in an appendix or short deployment-cost paragraph.",
        "- P1/P2 rows are best suited for appendix unless a reviewer asks specifically for them.",
    ]
    (ROOT / "extra_experiment_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    files = sorted(str(p) for p in ROOT.rglob("*") if p.is_file())
    (ROOT / "inventory.txt").write_text("\n".join(files) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-p1-p2", action="store_true", help="Only run P0 experiments.")
    parser.add_argument("--force-p0", action="store_true", help="Recompute P0 raw/GRAIN control and dependent low-FPR table.")
    args = parser.parse_args()
    setup_style()
    outputs: dict[str, pd.DataFrame] = {}
    if (TABLES / "raw_same_classifier_control.csv").exists() and not args.force_p0:
        print("[extra] reuse existing raw_same_classifier_control.csv", flush=True)
        outputs["raw_same_classifier_control"] = pd.read_csv(TABLES / "raw_same_classifier_control.csv")
    else:
        outputs["raw_same_classifier_control"] = run_raw_same_classifier_control()
    if (TABLES / "efficiency_cost.csv").exists():
        outputs["efficiency_cost"] = pd.read_csv(TABLES / "efficiency_cost.csv")
    else:
        outputs["efficiency_cost"] = run_efficiency_cost(outputs["raw_same_classifier_control"])
    if (TABLES / "cross_shift_matrix.csv").exists():
        outputs["cross_shift_matrix"] = pd.read_csv(TABLES / "cross_shift_matrix.csv")
    else:
        outputs["cross_shift_matrix"] = run_cross_shift_matrix(outputs["raw_same_classifier_control"])
    if (TABLES / "low_fpr_operating_curve.csv").exists() and not args.force_p0:
        outputs["low_fpr_operating_curve"] = pd.read_csv(TABLES / "low_fpr_operating_curve.csv")
    else:
        outputs["low_fpr_operating_curve"] = run_low_fpr_operating_curve(outputs["raw_same_classifier_control"])
    if not args.skip_p1_p2:
        if (TABLES / "feature_ablation_extended.csv").exists():
            outputs["feature_ablation_extended"] = pd.read_csv(TABLES / "feature_ablation_extended.csv")
        else:
            outputs["feature_ablation_extended"] = run_feature_ablation_extended()
        if (TABLES / "grain_classifier_sensitivity.csv").exists():
            outputs["grain_classifier_sensitivity"] = pd.read_csv(TABLES / "grain_classifier_sensitivity.csv")
        else:
            outputs["grain_classifier_sensitivity"] = run_classifier_sensitivity()
    write_report(outputs)
    print(f"[extra] done -> {ROOT}")


if __name__ == "__main__":
    main()
