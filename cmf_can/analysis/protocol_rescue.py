from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cluster import Birch
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


ROOT = Path(".")
RAW = ROOT / "data/raw/can-train-and-test/set_01"
OUT = ROOT / "results/protocol_rescue"
TABLES = OUT / "tables"
FIGS = OUT / "figures"
LOGS = OUT / "logs"
PREDS = OUT / "predictions"
CONFIGS = OUT / "configs"

TEST_FOLDERS = {
    "ctt_test01": "test_01_known_vehicle_known_attack",
    "ctt_test02": "test_02_unknown_vehicle_known_attack",
    "ctt_test03": "test_03_known_vehicle_unknown_attack",
    "ctt_test04": "test_04_unknown_vehicle_unknown_attack",
}

FEATURES = [
    "can_id",
    "dlc",
    "data0",
    "data1",
    "data2",
    "data3",
    "data4",
    "data5",
    "data6",
    "data7",
    "delta_t_global",
    "delta_t_same_id",
    "payload_sum",
    "payload_mean",
    "payload_std",
    "payload_delta_l1",
]


@dataclass
class EvalResult:
    dataset: str
    model: str
    precision: float
    recall: float
    f1: float
    auroc: float
    aupr: float
    fpr: float
    fnr: float
    threshold: float
    train_rows: int
    train_positive: int
    test_rows: int
    test_positive: int
    protocol: str


def setup() -> None:
    for p in [TABLES, FIGS, LOGS, PREDS, CONFIGS]:
        p.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def parse_payload(values: pd.Series) -> np.ndarray:
    texts = values.fillna("").astype(str).str.replace(" ", "", regex=False).str.upper().to_numpy()
    out = np.zeros((len(texts), 8), dtype=np.float32)
    for i, text in enumerate(texts):
        text = text[:16].ljust(16, "0")
        for j in range(8):
            try:
                out[i, j] = int(text[j * 2 : j * 2 + 2], 16) / 255.0
            except ValueError:
                out[i, j] = 0.0
    return out


def parse_can_id(values: pd.Series) -> np.ndarray:
    out = np.zeros(len(values), dtype=np.float32)
    for i, value in enumerate(values.astype(str)):
        text = value.strip()
        try:
            out[i] = int(text, 16) / 4095.0
        except ValueError:
            out[i] = float(value) / 4095.0
    return out


def read_ctt_file(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    payload = parse_payload(df["data_field"])
    can_id_raw = parse_can_id(df["arbitration_id"])
    ts = df["timestamp"].astype(float).to_numpy()
    delta_global = np.zeros(len(df), dtype=np.float32)
    if len(df) > 1:
        delta_global[1:] = np.maximum(np.diff(ts), 0).astype(np.float32)
    delta_same = np.zeros(len(df), dtype=np.float32)
    last_ts: dict[str, float] = {}
    last_payload: dict[str, np.ndarray] = {}
    payload_delta = np.zeros(len(df), dtype=np.float32)
    for i, (cid, t) in enumerate(zip(df["arbitration_id"].astype(str), ts)):
        if cid in last_ts:
            delta_same[i] = max(float(t - last_ts[cid]), 0.0)
            payload_delta[i] = float(np.abs(payload[i] - last_payload[cid]).sum()) / 8.0
        last_ts[cid] = float(t)
        last_payload[cid] = payload[i].copy()
    dlc = df["data_field"].fillna("").astype(str).str.replace(" ", "", regex=False).str.len().clip(0, 16).to_numpy(np.float32) / 16.0
    x = np.column_stack(
        [
            can_id_raw,
            dlc,
            payload,
            np.log1p(delta_global),
            np.log1p(delta_same),
            payload.sum(axis=1),
            payload.mean(axis=1),
            payload.std(axis=1),
            payload_delta,
        ]
    ).astype(np.float32)
    y = df["attack"].astype(int).clip(0, 1).to_numpy(np.int8)
    return x, y, df["arbitration_id"].astype(str).to_numpy()


def collect_train(max_neg_train: int = 60_000, max_neg_val: int = 30_000, seed: int = 42):
    rng = np.random.default_rng(seed)
    x_train, y_train, x_val, y_val = [], [], [], []
    for path in sorted((RAW / "train_01").glob("*.csv")):
        print(f"[protocol_rescue] read train {path}", flush=True)
        x, y, _ = read_ctt_file(path)
        is_val = path.stem.endswith("-2")
        pos = np.where(y == 1)[0]
        neg = np.where(y == 0)[0]
        cap = max_neg_val if is_val else max_neg_train
        if len(neg) > cap // 6:
            neg = rng.choice(neg, size=max(1, cap // 6), replace=False)
        idx = np.concatenate([pos, neg])
        rng.shuffle(idx)
        if is_val:
            x_val.append(x[idx]); y_val.append(y[idx])
        else:
            x_train.append(x[idx]); y_train.append(y[idx])
    return np.vstack(x_train), np.concatenate(y_train), np.vstack(x_val), np.concatenate(y_val)


def iter_test_folder(folder: str):
    for path in sorted((RAW / folder).glob("*.csv")):
        x, y, _ = read_ctt_file(path)
        yield path, x, y


def score_model(model, x: np.ndarray, kind: str) -> np.ndarray:
    if kind == "birch":
        centers = model.subcluster_centers_
        # chunk nearest-center distance to avoid a large dense distance matrix.
        out = np.empty(len(x), dtype=np.float32)
        for start in range(0, len(x), 50_000):
            part = x[start : start + 50_000]
            dist = ((part[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            out[start : start + 50_000] = np.sqrt(dist.min(axis=1))
        return out
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    if hasattr(model, "decision_function"):
        s = model.decision_function(x)
        return -s if kind == "isolation_forest" else s
    return model.predict(x).astype(float)


def threshold_from_val(y: np.ndarray, score: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y, score)
    if len(thresholds) == 0:
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.nanargmax(f1))])


def metrics(y: np.ndarray, score: np.ndarray, threshold: float) -> dict:
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
        "auroc": roc_auc_score(y, score) if len(np.unique(y)) == 2 else np.nan,
        "aupr": average_precision_score(y, score) if len(np.unique(y)) == 2 else np.nan,
        "fpr": fp / max(fp + tn, 1),
        "fnr": fn / max(fn + tp, 1),
    }


def train_models(x_train, y_train, x_val, y_val):
    scaler = StandardScaler().fit(x_train)
    xs = scaler.transform(x_train)
    xvs = scaler.transform(x_val)
    normal_xs = xs[y_train == 0]
    models = {
        "LogisticRegression": LogisticRegression(max_iter=200, class_weight="balanced"),
        "GaussianNB": GaussianNB(),
        "MLP": MLPClassifier(hidden_layer_sizes=(48,), max_iter=15, early_stopping=True, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=30, max_depth=2, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=30, max_depth=None, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=-1, random_state=42),
        "IsolationForest": IsolationForest(n_estimators=50, contamination="auto", random_state=42, n_jobs=-1),
        "BIRCH": Birch(threshold=2.0, n_clusters=None),
    }
    fitted = {}
    thresholds = {}
    for name, model in models.items():
        print(f"[protocol_rescue] fit {name}", flush=True)
        if name in {"IsolationForest", "BIRCH"}:
            fit_x = normal_xs
        else:
            fit_x = xs
        model.fit(fit_x, y_train if name not in {"IsolationForest", "BIRCH"} else None)
        kind = name.lower().replace("-", "_")
        val_score = score_model(model, xvs, "isolation_forest" if name == "IsolationForest" else "birch" if name == "BIRCH" else "supervised")
        fitted[name] = model
        thresholds[name] = threshold_from_val(y_val, val_score)
        print(f"[protocol_rescue] fitted {name} threshold={thresholds[name]:.6f}", flush=True)
    return scaler, fitted, thresholds


def official_ml_reproduction() -> pd.DataFrame:
    x_train, y_train, x_val, y_val = collect_train()
    scaler, fitted, thresholds = train_models(x_train, y_train, x_val, y_val)
    rows = []
    pred_dir = PREDS / "official_ml"
    pred_dir.mkdir(parents=True, exist_ok=True)
    for dataset, folder in TEST_FOLDERS.items():
        print(f"[protocol_rescue] eval {dataset}", flush=True)
        y_parts = []
        score_parts = {name: [] for name in fitted}
        for _, x, y in iter_test_folder(folder):
            print(f"[protocol_rescue] score chunk {dataset} rows={len(y)}", flush=True)
            xs = scaler.transform(x)
            y_parts.append(y)
            for name, model in fitted.items():
                kind = "isolation_forest" if name == "IsolationForest" else "birch" if name == "BIRCH" else "supervised"
                score_parts[name].append(score_model(model, xs, kind))
        y_all = np.concatenate(y_parts)
        for name, parts in score_parts.items():
            score = np.concatenate(parts)
            m = metrics(y_all, score, thresholds[name])
            rows.append(
                EvalResult(
                    dataset=dataset,
                    model=name,
                    threshold=thresholds[name],
                    train_rows=int(len(y_train)),
                    train_positive=int(y_train.sum()),
                    test_rows=int(len(y_all)),
                    test_positive=int(y_all.sum()),
                    protocol="official_sample_level_train_neg_sampled_test_full",
                    **m,
                ).__dict__
            )
            if name in {"RandomForest", "GradientBoosting", "LogisticRegression"}:
                pd.DataFrame({"label": y_all, "score": score}).to_csv(pred_dir / f"{dataset}_{name}_scores.csv", index=False)
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "ctt_official_ml_reproduction.csv", index=False)
    (TABLES / "ctt_official_ml_reproduction.tex").write_text(out.to_latex(index=False, escape=True, float_format=lambda x: f"{x:.4f}"), encoding="utf-8")
    plot_ml(out, "ctt_official_ml_reproduction", "Official sample-level ML")
    return out


def plot_ml(df: pd.DataFrame, name: str, title: str) -> None:
    best = df.copy()
    fig, ax = plt.subplots(figsize=(8, 3.4))
    pivot = best.pivot(index="dataset", columns="model", values="f1")
    x = np.arange(len(pivot.index))
    width = min(0.8 / max(len(pivot.columns), 1), 0.12)
    for i, col in enumerate(pivot.columns):
        ax.bar(x + (i - (len(pivot.columns) - 1) / 2) * width, pivot[col], width, label=col, edgecolor="#222")
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index)
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)
    ax.legend(frameon=False, ncol=4)
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(FIGS / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def granularity_comparison(official: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset in TEST_FOLDERS:
        best = official[official["dataset"].eq(dataset)].sort_values("f1", ascending=False).head(1)
        if not best.empty:
            rec = best.iloc[0].to_dict()
            rec["granularity"] = "official_sample_level"
            rows.append(rec)
    for path, protocol in [
        (ROOT / "results/cmf_tables/ctt_generalization_15ep.csv", "current_window_100_deep"),
        (ROOT / "results/cmf_tables/ctt_unknown_ablation.csv", "current_window_100_deep_ablation"),
    ]:
        if path.exists():
            df = pd.read_csv(path)
            for _, rec in df.iterrows():
                if rec["dataset"] in TEST_FOLDERS:
                    item = rec.to_dict()
                    item["granularity"] = protocol
                    rows.append(item)
    # Short windows are not materialized; keep explicit NA rows.
    for dataset in TEST_FOLDERS:
        for w in [10, 20, 50]:
            rows.append({"dataset": dataset, "model": "not_run", "granularity": f"short_window_{w}", "f1": np.nan, "note": "short-window processed datasets not built in this run"})
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "ctt_granularity_comparison.csv", index=False)
    (TABLES / "ctt_granularity_comparison.tex").write_text(out.to_latex(index=False, escape=True, na_rep="NA", float_format=lambda x: f"{x:.4f}"), encoding="utf-8")
    plot = out[pd.to_numeric(out.get("f1"), errors="coerce").notna()].copy()
    plot = plot.sort_values(["dataset", "f1"], ascending=[True, False]).groupby(["dataset", "granularity"]).head(1)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    labels = plot["dataset"].astype(str) + "\n" + plot["granularity"].astype(str)
    ax.bar(np.arange(len(plot)), plot["f1"].astype(float), edgecolor="#222")
    ax.set_xticks(np.arange(len(plot)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Best F1")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(FIGS / f"ctt_granularity_comparison.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def feature_importance(best_model_name: str = "RandomForest") -> pd.DataFrame:
    x_train, y_train, x_val, y_val = collect_train(max_neg_train=30_000, max_neg_val=15_000)
    scaler, fitted, _ = train_models(x_train, y_train, x_val, y_val)
    model = fitted.get(best_model_name) or fitted["GradientBoosting"]
    xs_val = scaler.transform(x_val)
    rows = []
    if hasattr(model, "feature_importances_"):
        for f, v in zip(FEATURES, model.feature_importances_):
            rows.append({"feature": f, "tree_importance": float(v)})
    else:
        for f in FEATURES:
            rows.append({"feature": f, "tree_importance": np.nan})
    base = pd.DataFrame(rows)
    aucs = []
    effects = []
    for i, f in enumerate(FEATURES):
        col = x_val[:, i]
        try:
            auc = roc_auc_score(y_val, col)
            auc = max(float(auc), 1.0 - float(auc))
        except ValueError:
            auc = np.nan
        pos = col[y_val == 1]
        neg = col[y_val == 0]
        pooled = math.sqrt(float(pos.var() + neg.var()) / 2.0 + 1e-12) if len(pos) and len(neg) else np.nan
        effect = float((pos.mean() - neg.mean()) / pooled) if pooled and not np.isnan(pooled) else np.nan
        aucs.append({"feature": f, "univariate_auc_abs": auc})
        effects.append({"feature": f, "effect_size": effect})
    out = base.merge(pd.DataFrame(aucs), on="feature").merge(pd.DataFrame(effects), on="feature")
    out = out.sort_values(["tree_importance", "univariate_auc_abs"], ascending=False)
    out.to_csv(TABLES / "ctt_feature_importance.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    top = out.head(12)
    ax.barh(top["feature"][::-1], top["tree_importance"][::-1], edgecolor="#222")
    ax.set_xlabel("Tree feature importance")
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.7)
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(FIGS / f"ctt_feature_importance.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def event_level_from_predictions() -> pd.DataFrame:
    rows = []
    pred_sources = [
        ("road", "Transformer", ROOT / "results/cmf_predictions/road_transformer_predictions.csv"),
        ("road", "CAN-Transformer+ same-ID", ROOT / "results/transformer_rescue/predictions/road_can_transformer_plus_sameid_predictions.csv"),
        ("ctt_test02", "Transformer", ROOT / "results/cmf_predictions/ctt_test02_transformer_predictions.csv"),
        ("ctt_test02", "CMF-CAN", ROOT / "results/cmf_predictions/ctt_test02_cmf_can_predictions.csv"),
        ("ctt_test04", "Transformer", ROOT / "results/cmf_predictions/ctt_test04_transformer_predictions.csv"),
        ("ctt_test04", "CMF-CAN", ROOT / "results/cmf_predictions/ctt_test04_cmf_can_predictions.csv"),
    ]
    for dataset, model, path in pred_sources:
        if not path.exists():
            continue
        df = pd.read_csv(path).sort_values("window_start")
        y = df["label"].astype(int).to_numpy()
        pred = df["prediction"].astype(int).to_numpy()
        starts = df["window_start"].to_numpy()
        # contiguous positive-label windows form a coarse event.
        event_starts = np.where((y == 1) & np.r_[True, y[:-1] == 0])[0]
        event_ends = np.r_[event_starts[1:], len(y)]
        detected = 0
        delays = []
        for s, e in zip(event_starts, event_ends):
            hits = np.where(pred[s:e] == 1)[0]
            if len(hits):
                detected += 1
                delays.append(int(starts[s + hits[0]] - starts[s]))
        fp_windows = int(((pred == 1) & (y == 0)).sum())
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "events": int(len(event_starts)),
                "event_recall": detected / max(len(event_starts), 1),
                "mean_detection_delay_windows": float(np.mean(delays)) if delays else np.nan,
                "median_detection_delay_windows": float(np.median(delays)) if delays else np.nan,
                "false_alarm_windows": fp_windows,
                "false_alarm_windows_per_100k_windows": fp_windows / max(len(y), 1) * 100_000,
                "note": "events are contiguous positive windows; no official event boundaries available",
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "event_level_metrics.csv", index=False)
    if not out.empty:
        (TABLES / "event_level_metrics.tex").write_text(out.to_latex(index=False, escape=True, na_rep="NA", float_format=lambda x: f"{x:.4f}"), encoding="utf-8")
        fig, ax = plt.subplots(figsize=(7, 3.2))
        labels = out["dataset"].astype(str) + "\n" + out["model"].astype(str)
        ax.bar(np.arange(len(out)), out["event_recall"], edgecolor="#222")
        ax.set_xticks(np.arange(len(out)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("Event recall")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)
        for ext in ["png", "pdf", "svg"]:
            fig.savefig(FIGS / f"event_level_metrics.{ext}", dpi=300, bbox_inches="tight")
        plt.close(fig)
    return out


def normality_policy() -> pd.DataFrame:
    rows = []
    for dataset in ["ctt_test03", "ctt_test04", "road"]:
        path = ROOT / f"results/cmf_predictions/{dataset}_transformer_predictions.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        y = df["label"].astype(int).to_numpy()
        classifier = df["score"].astype(float).to_numpy()
        # Simple post-hoc normality score from score tail distance; no training leakage.
        base_scores = classifier[y == 0] if (y == 0).any() else classifier
        normality = np.abs(classifier - np.median(base_scores))
        for name, score in [
            ("classifier_score_only", classifier),
            ("normality_score_only", normality),
            ("max_classifier_normality", np.maximum(classifier, normality)),
            ("half_classifier_half_normality", 0.5 * classifier + 0.5 * normality),
        ]:
            t = threshold_from_val(y, score)  # upper bound because validation predictions are unavailable.
            m = metrics(y, score, t)
            rows.append({"dataset": dataset, "policy": name, "threshold_source": "test_upper_bound_no_val_dump", **m})
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "normality_policy_results.csv", index=False)
    return out


def write_analyses(official: pd.DataFrame, gran: pd.DataFrame, imp: pd.DataFrame, events: pd.DataFrame, norm: pd.DataFrame) -> None:
    best = official.sort_values("f1", ascending=False).groupby("dataset").head(1)
    best_text = best[["dataset", "model", "f1", "aupr", "auroc", "fpr", "fnr"]].to_csv(index=False)
    top_imp_text = imp.head(10).to_csv(index=False)
    (OUT / "ctt_official_ml_reproduction_analysis.md").write_text(
        "# CT&T Official ML Reproduction Analysis\n\n"
        "This run uses official CT&T frame-level CSVs and does not use the current window=100 deep-learning pipeline. Training keeps all sampled positives and a capped random subset of negatives; testing is full official test folders.\n\n"
        f"Best per setting:\n\n```csv\n{best_text}```\n\n"
        "Findings:\n"
        "1. If test02/test04 are high here, the protocol gap is caused by the window pipeline rather than data absence.\n"
        "2. If test04 remains low, the public benchmark likely depends on different fields, split assumptions, or full training protocol details.\n"
        "3. Because training negatives are sampled, this is a protocol rescue pilot, not a strict full-benchmark reproduction.\n",
        encoding="utf-8",
    )
    (OUT / "ctt_granularity_comparison_analysis.md").write_text(
        "# CT&T Granularity Comparison Analysis\n\n"
        "Official sample-level ML is compared with existing window=100 deep-learning results. Short-window datasets were not materialized in this run and are explicitly marked NA.\n\n"
        "If sample-level ML beats window=100 on shifted settings, the current window pipeline is losing or diluting discriminative per-frame signals.\n",
        encoding="utf-8",
    )
    (OUT / "ctt_feature_importance_analysis.md").write_text(
        "# CT&T Feature Importance Analysis\n\n"
        f"Top features:\n\n```csv\n{top_imp_text}```\n\n"
        "High importance on CAN ID, timing gaps, or payload deltas means these fields must be preserved at frame/short-window granularity. Any feature that directly encodes attack labels would be leakage; this script uses only timestamp, arbitration_id, data_field-derived bytes/DLC, and derived temporal deltas.\n",
        encoding="utf-8",
    )
    (OUT / "event_level_metrics_analysis.md").write_text(
        "# Event-Level Metrics Analysis\n\n"
        "No official event boundary files were found. Events here are approximated as contiguous positive windows ordered by window_start, so delay is reported in frame-index units and should be treated as coarse evidence only.\n",
        encoding="utf-8",
    )
    (OUT / "normality_policy_analysis.md").write_text(
        "# Normality Policy Analysis\n\n"
        "This run only evaluates a post-hoc score-combination pilot because validation prediction dumps for all candidates are unavailable. Thresholds are marked as test upper bounds and must not be used as formal main results.\n",
        encoding="utf-8",
    )
    (OUT / "final_direction_decision.md").write_text(
        "# Final Direction Decision\n\n"
        "1. CAN-Transformer+ is unlikely to directly exceed Transformer on ROAD under the current window=100 protocol.\n"
        "2. CT&T official frame-level protocol is the key next evidence source; if its ML baselines recover strong shifted performance, the deep window pipeline is the bottleneck.\n"
        "3. window=100 should not be the default for shifted CT&T until short-window experiments show otherwise.\n"
        "4. The most promising next direction is feature-preserving sample/short-window detection plus event-level/low-false-alarm evaluation, not full multimodal fusion.\n"
        "5. For CCF A/Security Four, the viable story is protocol-correct, deployment-level CAN IDS under shift; the current end-to-end deep window classifier alone is not enough.\n"
        "6. If full official-feature ML and short-window protocols also fail to beat Transformer/public baselines, the current topic should be stopped or reframed as a negative protocol study.\n",
        encoding="utf-8",
    )


def main() -> None:
    setup()
    (CONFIGS / "protocol_rescue_config.json").write_text(
        json.dumps(
            {
                "raw_root": str(RAW),
                "train_negative_sampling": True,
                "max_neg_train_total_approx": 500000,
                "max_neg_val_total_approx": 200000,
                "test_protocol": "full official test folders",
                "features": FEATURES,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    official = official_ml_reproduction()
    gran = granularity_comparison(official)
    imp = feature_importance()
    events = event_level_from_predictions()
    norm = normality_policy()
    write_analyses(official, gran, imp, events, norm)
    print("[protocol_rescue] done")


if __name__ == "__main__":
    main()
