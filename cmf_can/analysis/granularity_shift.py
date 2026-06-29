from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import Birch
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from cmf_can.analysis.protocol_rescue import (
    FEATURES,
    RAW,
    TEST_FOLDERS,
    collect_train,
    iter_test_folder,
    metrics,
    read_ctt_file,
    score_model,
    threshold_from_val,
)


ROOT = Path(".")
OUT = ROOT / "results/granularity_shift"
TABLES = OUT / "tables"
FIGS = OUT / "figures"
LOGS = OUT / "logs"
PREDS = OUT / "predictions"
CONFIGS = OUT / "configs"
SEEDS = [42, 2024, 2026, 7, 99]


def recall_at_fpr(y: np.ndarray, score: np.ndarray, budget: float) -> float:
    order = np.argsort(-score)
    y_sorted = y[order]
    pos = max(int((y == 1).sum()), 1)
    neg = max(int((y == 0).sum()), 1)
    tp = 0
    fp = 0
    best = 0.0
    for label in y_sorted:
        if int(label) == 1:
            tp += 1
        else:
            fp += 1
        if fp / neg <= budget:
            best = tp / pos
        else:
            break
    return float(best)


def setup() -> None:
    for path in [TABLES, FIGS, LOGS, PREDS, CONFIGS]:
        path.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def model_bank(seed: int):
    return {
        "GradientBoosting": GradientBoostingClassifier(n_estimators=35, max_depth=2, random_state=seed),
        "RandomForest": RandomForestClassifier(n_estimators=35, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=-1, random_state=seed),
        "MLP": MLPClassifier(hidden_layer_sizes=(48,), max_iter=15, early_stopping=True, random_state=seed),
        "LogisticRegression": LogisticRegression(max_iter=200, class_weight="balanced"),
        "GaussianNB": GaussianNB(),
        "IsolationForest": IsolationForest(n_estimators=50, contamination="auto", n_jobs=-1, random_state=seed),
        "BIRCH": Birch(threshold=2.0, n_clusters=None),
    }


def fit_models(seed: int, neg_train: int = 60_000, neg_val: int = 30_000):
    x_train, y_train, x_val, y_val = collect_train(max_neg_train=neg_train, max_neg_val=neg_val, seed=seed)
    scaler = StandardScaler().fit(x_train)
    xs = scaler.transform(x_train)
    xvs = scaler.transform(x_val)
    normal_xs = xs[y_train == 0]
    fitted = {}
    thresholds = {}
    for name, model in model_bank(seed).items():
        print(f"[granularity_shift] fit seed={seed} model={name}", flush=True)
        if name in {"IsolationForest", "BIRCH"}:
            model.fit(normal_xs)
            kind = "isolation_forest" if name == "IsolationForest" else "birch"
        else:
            model.fit(xs, y_train)
            kind = "supervised"
        score = score_model(model, xvs, kind)
        fitted[name] = model
        thresholds[name] = threshold_from_val(y_val, score)
    return scaler, fitted, thresholds, (x_train, y_train, x_val, y_val)


def samplelevel_5seed() -> pd.DataFrame:
    rows = []
    saved_pred = set()
    for seed in SEEDS:
        scaler, fitted, thresholds, _ = fit_models(seed)
        for dataset, folder in TEST_FOLDERS.items():
            print(f"[granularity_shift] eval sample seed={seed} dataset={dataset}", flush=True)
            y_parts = []
            scores = {name: [] for name in fitted}
            for _, x, y in iter_test_folder(folder):
                xs = scaler.transform(x)
                y_parts.append(y)
                for name, model in fitted.items():
                    kind = "isolation_forest" if name == "IsolationForest" else "birch" if name == "BIRCH" else "supervised"
                    scores[name].append(score_model(model, xs, kind))
            y_all = np.concatenate(y_parts)
            for name, parts in scores.items():
                score = np.concatenate(parts)
                m = metrics(y_all, score, thresholds[name])
                rows.append({
                    "dataset": dataset,
                    "model": name,
                    "seed": seed,
                    "granularity": "sample",
                    "threshold": thresholds[name],
                    "train_protocol": "all positives + capped negative sampling",
                    "test_protocol": "full official test folder",
                    **m,
                })
                key = (name, dataset)
                if seed == 42 and name in {"GradientBoosting", "RandomForest"} and key not in saved_pred:
                    # Store a deterministic compact prediction audit, not full 10M+ frame dumps.
                    rng = np.random.default_rng(42)
                    n = len(y_all)
                    keep = np.arange(n) if n <= 50_000 else np.sort(rng.choice(n, size=50_000, replace=False))
                    pred = (score[keep] >= thresholds[name]).astype(int)
                    pd.DataFrame({
                        "sample_index": keep,
                        "dataset": dataset,
                        "model": name,
                        "seed": seed,
                        "label": y_all[keep],
                        "score": score[keep],
                        "prediction": pred,
                        "threshold": thresholds[name],
                        "sampling_note": "deterministic 50k audit sample; full dumps omitted due size",
                    }).to_csv(PREDS / f"{name}_{dataset}_samplelevel_predictions.csv", index=False)
                    saved_pred.add(key)
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "ctt_official_samplelevel_5seed.csv", index=False)
    (TABLES / "ctt_official_samplelevel_5seed.tex").write_text(
        out.to_latex(index=False, escape=True, float_format=lambda x: f"{x:.4f}"), encoding="utf-8"
    )
    plot_5seed(out)
    return out


def plot_5seed(df: pd.DataFrame) -> None:
    best = df.groupby(["dataset", "model"], as_index=False).agg(f1_mean=("f1", "mean"), f1_std=("f1", "std"))
    best = best.sort_values("f1_mean", ascending=False).groupby("dataset").head(3)
    fig, ax = plt.subplots(figsize=(8, 3.4))
    labels = best["dataset"] + "\n" + best["model"]
    ax.bar(np.arange(len(best)), best["f1_mean"], yerr=best["f1_std"].fillna(0), edgecolor="#222", capsize=2)
    ax.set_xticks(np.arange(len(best)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("F1 mean ± std")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(FIGS / f"ctt_official_samplelevel_5seed.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def window_features(x: np.ndarray, y: np.ndarray, w: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(y) // w
    if n == 0:
        return np.empty((0, 0)), np.empty((0,), dtype=int), np.empty((0,), dtype=float)
    xx = x[: n * w].reshape(n, w, x.shape[1])
    yy = y[: n * w].reshape(n, w)
    last = xx[:, -1, :]
    mean = xx.mean(axis=1)
    std = xx.std(axis=1)
    mx = xx.max(axis=1)
    mn = xx.min(axis=1)
    attack_ratio = yy.mean(axis=1)
    label = (attack_ratio > 0).astype(int)
    return np.concatenate([last, mean, std, mx, mn], axis=1), label, attack_ratio


def collect_window_train(w: int, seed: int = 42, max_neg: int = 80_000):
    rng = np.random.default_rng(seed)
    xs_train, ys_train, xs_val, ys_val = [], [], [], []
    for path in sorted((RAW / "train_01").glob("*.csv")):
        x, y, _ = read_ctt_file(path)
        wx, wy, _ = window_features(x, y, w)
        if len(wy) == 0:
            continue
        pos = np.where(wy == 1)[0]
        neg = np.where(wy == 0)[0]
        cap = max(1, max_neg // 6)
        if len(neg) > cap:
            neg = rng.choice(neg, size=cap, replace=False)
        idx = np.concatenate([pos, neg])
        rng.shuffle(idx)
        if path.stem.endswith("-2"):
            xs_val.append(wx[idx]); ys_val.append(wy[idx])
        else:
            xs_train.append(wx[idx]); ys_train.append(wy[idx])
    return np.vstack(xs_train), np.concatenate(ys_train), np.vstack(xs_val), np.concatenate(ys_val)


def eval_window_model(model, scaler, threshold, w: int, dataset: str, folder: str):
    y_parts, s_parts, ratio_parts = [], [], []
    for _, x, y in iter_test_folder(folder):
        wx, wy, ar = window_features(x, y, w)
        if len(wy) == 0:
            continue
        score = model.predict_proba(scaler.transform(wx))[:, 1]
        y_parts.append(wy); s_parts.append(score); ratio_parts.append(ar)
    y_all = np.concatenate(y_parts)
    score_all = np.concatenate(s_parts)
    ar_all = np.concatenate(ratio_parts)
    return y_all, score_all, ar_all


def granularity_search() -> pd.DataFrame:
    rows = []
    sample = pd.read_csv(TABLES / "ctt_official_samplelevel_5seed.csv")
    for _, rec in sample.groupby(["dataset", "model"], as_index=False).agg(f1=("f1", "mean"), auroc=("auroc", "mean"), aupr=("aupr", "mean")).iterrows():
        if rec["model"] == "GradientBoosting":
            rows.append({"dataset": rec["dataset"], "model": "GradientBoosting", "granularity": "sample", "window_size": 1, "f1": rec["f1"], "auroc": rec["auroc"], "aupr": rec["aupr"]})
    for w in [5, 10, 20, 50, 100]:
        print(f"[granularity_shift] train short window w={w}", flush=True)
        x_train, y_train, x_val, y_val = collect_window_train(w)
        scaler = StandardScaler().fit(x_train)
        model = GradientBoostingClassifier(n_estimators=35, max_depth=2, random_state=42)
        model.fit(scaler.transform(x_train), y_train)
        val_score = model.predict_proba(scaler.transform(x_val))[:, 1]
        threshold = threshold_from_val(y_val, val_score)
        for dataset, folder in TEST_FOLDERS.items():
            y, score, ar = eval_window_model(model, scaler, threshold, w, dataset, folder)
            m = metrics(y, score, threshold)
            rows.append({"dataset": dataset, "model": "GradientBoosting", "granularity": f"window_{w}", "window_size": w, "mean_attack_ratio": float(ar[y == 1].mean()) if (y == 1).any() else np.nan, **m})
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "granularity_search.csv", index=False)
    (TABLES / "granularity_search.tex").write_text(out.to_latex(index=False, escape=True, na_rep="NA", float_format=lambda x: f"{x:.4f}"), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(8, 3.4))
    plot = out[out["dataset"].isin(["ctt_test02", "ctt_test04"])].copy()
    for ds, g in plot.groupby("dataset"):
        g = g.sort_values("window_size")
        ax.plot(g["window_size"], g["f1"], marker="o", label=ds)
    ax.set_xscale("log")
    ax.set_xlabel("Window size (1 = sample)")
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1.05)
    ax.grid(color="#E5E7EB", linewidth=0.7)
    ax.legend(frameon=False)
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(FIGS / f"granularity_search.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def normality_policy() -> pd.DataFrame:
    seed = 42
    scaler, fitted, thresholds, train_pack = fit_models(seed)
    x_train, y_train, x_val, y_val = train_pack
    gb = fitted["GradientBoosting"]
    normal = x_train[y_train == 0]
    med = np.median(normal, axis=0)
    mad = np.median(np.abs(normal - med), axis=0) + 1e-6
    val_sup = gb.predict_proba(scaler.transform(x_val))[:, 1]
    val_norm = np.clip(np.abs((x_val - med) / mad).mean(axis=1) / 10.0, 0, 1)
    candidates = []
    for alpha in np.linspace(0, 1, 11):
        score = alpha * val_sup + (1 - alpha) * val_norm
        t = threshold_from_val(y_val, score)
        candidates.append((alpha, t, metrics(y_val, score, t)["f1"]))
    alpha, threshold, _ = max(candidates, key=lambda x: x[2])
    rows = []
    for dataset in ["ctt_test03", "ctt_test04"]:
        y_parts, sup_parts, norm_parts = [], [], []
        for _, x, y in iter_test_folder(TEST_FOLDERS[dataset]):
            y_parts.append(y)
            sup_parts.append(gb.predict_proba(scaler.transform(x))[:, 1])
            norm_parts.append(np.clip(np.abs((x - med) / mad).mean(axis=1) / 10.0, 0, 1))
        y = np.concatenate(y_parts)
        sup = np.concatenate(sup_parts)
        norm = np.concatenate(norm_parts)
        policies = {
            "supervised_only": (sup, thresholds["GradientBoosting"]),
            "normality_only": (norm, threshold_from_val(y_val, val_norm)),
            "max_supervised_normality": (np.maximum(sup, norm), threshold),
            "val_selected_alpha_mix": (alpha * sup + (1 - alpha) * norm, threshold),
        }
        for name, (score, t) in policies.items():
            m = metrics(y, score, t)
            rows.append({
                "dataset": dataset,
                "policy": name,
                "alpha_supervised": alpha if "mix" in name else "",
                "threshold": t,
                **m,
                "recall_at_fpr_1em04": recall_at_fpr(y, score, 1e-4),
                "recall_at_fpr_5em04": recall_at_fpr(y, score, 5e-4),
                "recall_at_fpr_1em03": recall_at_fpr(y, score, 1e-3),
            })
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "test04_normality_policy.csv", index=False)
    (TABLES / "test04_normality_policy.tex").write_text(out.to_latex(index=False, escape=True, na_rep="NA", float_format=lambda x: f"{x:.4f}"), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(6, 3.2))
    plot = out[out["dataset"].eq("ctt_test04")]
    ax.bar(plot["policy"], plot["f1"], edgecolor="#222")
    ax.tick_params(axis="x", rotation=30)
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(FIGS / f"test04_normality_policy.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def event_level() -> pd.DataFrame:
    # Use available window-level prediction files; official sample-level event boundaries are not provided.
    rows = []
    sources = [
        ("ctt_test02", "window100_transformer", ROOT / "results/cmf_predictions/ctt_test02_transformer_predictions.csv"),
        ("ctt_test04", "window100_transformer", ROOT / "results/cmf_predictions/ctt_test04_transformer_predictions.csv"),
        ("road", "window100_transformer", ROOT / "results/cmf_predictions/road_transformer_predictions.csv"),
        ("road", "can_transformer_plus_sameid", ROOT / "results/transformer_rescue/predictions/road_can_transformer_plus_sameid_predictions.csv"),
    ]
    for dataset, model, path in sources:
        if not path.exists():
            continue
        df = pd.read_csv(path).sort_values("window_start")
        y = df["label"].astype(int).to_numpy()
        p = df["prediction"].astype(int).to_numpy()
        event_start = np.where((y == 1) & np.r_[True, y[:-1] == 0])[0]
        event_end = np.r_[event_start[1:], len(y)]
        hit = 0
        delays = []
        for s, e in zip(event_start, event_end):
            idx = np.where(p[s:e] == 1)[0]
            if len(idx):
                hit += 1
                delays.append(int(idx[0]))
        fp = int(((p == 1) & (y == 0)).sum())
        rows.append({
            "dataset": dataset,
            "model": model,
            "event_recall": hit / max(len(event_start), 1),
            "events": int(len(event_start)),
            "mean_detection_delay_windows": float(np.mean(delays)) if delays else np.nan,
            "median_detection_delay_windows": float(np.median(delays)) if delays else np.nan,
            "false_alarm_windows": fp,
            "false_alarm_windows_per_100k": fp / max(len(y), 1) * 100_000,
            "note": "contiguous positive windows; no official event boundary files",
        })
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "event_level_low_fa.csv", index=False)
    (TABLES / "event_level_low_fa.tex").write_text(out.to_latex(index=False, escape=True, na_rep="NA", float_format=lambda x: f"{x:.4f}"), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(6, 3.2))
    labels = out["dataset"] + "\n" + out["model"]
    ax.bar(np.arange(len(out)), out["event_recall"], edgecolor="#222")
    ax.set_xticks(np.arange(len(out)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Event recall")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(FIGS / f"event_level_low_fa.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def write_reports(sample: pd.DataFrame, gran: pd.DataFrame, norm: pd.DataFrame, event: pd.DataFrame) -> None:
    summary = sample.groupby(["dataset", "model"], as_index=False).agg(
        f1_mean=("f1", "mean"), f1_std=("f1", "std"), aupr_mean=("aupr", "mean"), auroc_mean=("auroc", "mean"), fpr_mean=("fpr", "mean")
    )
    best = summary.sort_values("f1_mean", ascending=False).groupby("dataset").head(1)
    (OUT / "ctt_official_samplelevel_analysis.md").write_text(
        "# CT&T Official Sample-Level 5-Seed Analysis\n\n"
        f"Best per setting:\n\n```csv\n{best.to_csv(index=False)}```\n\n"
        "- test02 is considered solved at sample level if the 5-seed mean remains near the protocol_rescue pilot F1=0.9662.\n"
        "- test04 remains unresolved unless its best mean clearly exceeds the previous GradientBoosting F1=0.3332.\n"
        "- Prediction CSVs are deterministic 50k audit samples for seed 42 GradientBoosting/RandomForest; full per-frame dumps were omitted due size.\n",
        encoding="utf-8",
    )
    gbest = gran.sort_values("f1", ascending=False).groupby("dataset").head(3)
    (OUT / "granularity_search_analysis.md").write_text(
        "# Granularity Search Analysis\n\n"
        f"Top rows:\n\n```csv\n{gbest.to_csv(index=False)}```\n\n"
        "The search uses feature-preserving GradientBoosting window aggregates. It stores any-attack labels and attack-ratio summaries. If window=100 drops relative to sample/short-window, it should not be the default shifted-CT&T protocol.\n",
        encoding="utf-8",
    )
    (OUT / "test04_normality_policy_analysis.md").write_text(
        "# Test04 Normality Policy Analysis\n\n"
        f"Results:\n\n```csv\n{norm.to_csv(index=False)}```\n\n"
        "Normality is train-normal only via robust median/MAD feature deviations. Alpha is selected on validation, not test04. If test04 F1/AUPR do not beat supervised-only GradientBoosting, unknown attack remains unsolved.\n",
        encoding="utf-8",
    )
    (OUT / "event_level_low_fa_analysis.md").write_text(
        "# Event-Level Low False Alarm Analysis\n\n"
        f"Results:\n\n```csv\n{event.to_csv(index=False)}```\n\n"
        "No official event boundaries were found. Events are approximated as contiguous positive windows; false alarms are reported per 100k windows rather than per hour.\n",
        encoding="utf-8",
    )
    (OUT / "feature_leakage_audit.md").write_text(
        "# Feature and Leakage Audit\n\n"
        "1. `delta_t_same_id` is computed from previous timestamp of the same CAN ID only.\n"
        "2. `payload_delta_l1` is computed from previous payload of the same CAN ID only.\n"
        "3. StandardScaler and robust normality median/MAD are fitted on train data only.\n"
        "4. Transition/profile features beyond past deltas were not added in this run; no test-derived transition profile is used.\n"
        "5. Test labels are used only for evaluation metrics, not feature construction or threshold selection.\n"
        "6. Sample-level features use timestamp, arbitration_id, data_field-derived bytes/DLC, and past-only deltas; no direct label field is included.\n"
        "7. Negative sampling affects training variance, so 5 seeds are reported. Full negative training was not completed because CT&T train_01 has over 10M frames.\n",
        encoding="utf-8",
    )
    best_method = best.sort_values("f1_mean", ascending=False).head(1)
    (OUT / "final_ccfa_security4_decision.md").write_text(
        "# Final CCF-A / Security-Four Direction Decision\n\n"
        f"Best observed sample-level row:\n\n```csv\n{best_method.to_csv(index=False)}```\n\n"
        "1. The strongest current direction is official-feature sample/short-window GradientBoosting, not CMF/Reliable-CMF/TFS gate.\n"
        "2. ROAD Transformer is still not beaten by CAN-Transformer+ under window=100; shifted CT&T test02 is beaten by official sample-level ML.\n"
        "3. test02 is solved only under feature-preserving sample/short-window protocol, not under window=100 deep models.\n"
        "4. test04 requires judging the normality table: if it does not improve over sample-level GradientBoosting, unknown attack remains the key gap.\n"
        "5. This has CCF A/Security-Four potential only if reframed as protocol-correct shifted CAN IDS with event/low-false-alarm guarantees; otherwise it is CCF B or a protocol-gap study.\n"
        "6. Next external work: real Mamba/SSM or external vehicle data is useful only after short-window/event-level protocol is fixed.\n",
        encoding="utf-8",
    )


def main() -> None:
    setup()
    (CONFIGS / "granularity_shift_config.json").write_text(json.dumps({
        "seeds": SEEDS,
        "features": FEATURES,
        "samplelevel_negative_sampling": "all positives + capped negatives per seed",
        "prediction_dump_policy": "50k deterministic audit samples only",
        "window_sizes": [5, 10, 20, 50, 100],
    }, indent=2), encoding="utf-8")
    sample = samplelevel_5seed()
    gran = granularity_search()
    norm = normality_policy()
    event = event_level()
    write_reports(sample, gran, norm, event)
    print("[granularity_shift] done", flush=True)


if __name__ == "__main__":
    main()
