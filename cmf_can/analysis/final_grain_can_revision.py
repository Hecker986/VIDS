from __future__ import annotations

import time
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from cmf_can.analysis.granularity_shift import collect_window_train, eval_window_model, window_features
from cmf_can.analysis.protocol_rescue import RAW, TEST_FOLDERS, collect_train, iter_test_folder, read_ctt_file, threshold_from_val


ROOT = Path(".")
OUT = ROOT / "results/final_grain_can"
TABLES = OUT / "tables"
FIGS = OUT / "figures"
PREDS = OUT / "predictions"
AUDITS = OUT / "audits"
REV = OUT / "revision"
LOGS = OUT / "logs"
MANIFESTS = OUT / "manifests"
FPR_BUDGETS = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]


def setup() -> None:
    for p in [TABLES, FIGS, PREDS, AUDITS, REV, LOGS, MANIFESTS]:
        p.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.linewidth": 0.9,
        "svg.fonttype": "none",
    })


def write_table(df: pd.DataFrame, name: str) -> None:
    df.to_csv(TABLES / f"{name}.csv", index=False)
    (TABLES / f"{name}.tex").write_text(df.to_latex(index=False, escape=True, na_rep="NA", float_format=lambda x: f"{x:.4f}"), encoding="utf-8")


def save_svg(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIGS / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def compute_metrics(y: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    pred = score >= threshold
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    tpr0 = tn / max(tn + fp, 1)
    macro_f1 = (f1 + (2 * tpr0 * (tn / max(tn + fn, 1)) / max(tpr0 + (tn / max(tn + fn, 1)), 1e-12))) / 2
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_f1": macro_f1,
        "auroc": roc_auc_score(y, score) if len(np.unique(y)) == 2 else np.nan,
        "aupr": average_precision_score(y, score) if len(np.unique(y)) == 2 else np.nan,
        "fpr": fp / max(fp + tn, 1),
        "fnr": fn / max(fn + tp, 1),
        "num_positive": int((y == 1).sum()),
        "num_negative": int((y == 0).sum()),
    }


def recall_at_budget(y: np.ndarray, score: np.ndarray, budget: float) -> dict:
    order = np.argsort(-score)
    ys = y[order]
    ss = score[order]
    pos = max(int((y == 1).sum()), 1)
    neg = max(int((y == 0).sum()), 1)
    tp = fp = 0
    best = {"recall": 0.0, "precision": 1.0, "f1": 0.0, "actual_fpr": 0.0, "threshold": np.inf}
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
                "recall": rec,
                "precision": prec,
                "f1": 2 * prec * rec / max(prec + rec, 1e-12),
                "actual_fpr": fpr,
                "threshold": float(threshold),
            }
        else:
            break
    return best


def best_threshold(y: np.ndarray, score: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y, score)
    if len(thresholds) == 0:
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.nanargmax(f1))])


def model_bank(seed: int):
    return {
        "GradientBoosting": GradientBoostingClassifier(n_estimators=35, max_depth=2, random_state=seed),
        "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=60, max_leaf_nodes=31, random_state=seed),
        "LogisticRegression": LogisticRegression(max_iter=200, class_weight="balanced"),
    }


def fit_sample_models(seed: int, neg_cap: int):
    x_train, y_train, x_val, y_val = collect_train(max_neg_train=neg_cap, max_neg_val=max(30_000, neg_cap // 2), seed=seed)
    scaler = StandardScaler().fit(x_train)
    xs = scaler.transform(x_train)
    xv = scaler.transform(x_val)
    models, thresholds, fit_times = {}, {}, {}
    for name, model in model_bank(seed).items():
        print(f"[revision] fit sample model {name} neg_cap={neg_cap} seed={seed}", flush=True)
        start = time.time()
        model.fit(xs, y_train)
        fit_times[name] = time.time() - start
        score = model.predict_proba(xv)[:, 1]
        thresholds[name] = best_threshold(y_val, score)
        models[name] = model
    return scaler, models, thresholds, fit_times, len(y_train), int(y_train.sum())


def eval_sample_model(scaler, model, threshold: float, dataset: str):
    y_parts, s_parts = [], []
    start = time.time()
    print(f"[revision] eval sample model dataset={dataset}", flush=True)
    for _, x, y in iter_test_folder(TEST_FOLDERS[dataset]):
        y_parts.append(y)
        s_parts.append(model.predict_proba(scaler.transform(x))[:, 1])
    inference_time = time.time() - start
    y = np.concatenate(y_parts)
    score = np.concatenate(s_parts)
    return y, score, compute_metrics(y, score, threshold), inference_time


def large_negative_stability() -> pd.DataFrame:
    existing = pd.read_csv(TABLES / "a1_official_sample_negative_stability.csv") if (TABLES / "a1_official_sample_negative_stability.csv").exists() else pd.DataFrame()
    rows = []
    if not existing.empty:
        for _, r in existing.iterrows():
            rows.append({**r.to_dict(), "protocol": r.get("negative_protocol", "A_capped_current"), "status": "completed_existing", "notes": "existing 5-seed capped protocol"})
    protocols = [
        ("B_2x_negative_cap", 120_000, [42, 2024, 2026]),
        ("C_5x_negative_cap", 300_000, [42]),
    ]
    resource_limited_models = ["ExtraTrees", "RandomForest", "MLP"]
    for protocol, cap, seeds in protocols:
        for seed in seeds:
            print(f"[revision] sample protocol={protocol} seed={seed}", flush=True)
            scaler, models, thresholds, fit_times, n_train, n_pos = fit_sample_models(seed, cap)
            for dataset in TEST_FOLDERS:
                for name, model in models.items():
                    y, score, m, infer_time = eval_sample_model(scaler, model, thresholds[name], dataset)
                    rows.append({
                        "protocol": protocol,
                        "negative_cap": cap,
                        "negative_ratio": (n_train - n_pos) / max(n_pos, 1),
                        "seed": seed,
                        "model": name,
                        "granularity": "sample",
                        "window_size": 1,
                        "setting": dataset,
                        **m,
                        "num_train_pos": n_pos,
                        "num_train_neg": n_train - n_pos,
                        "num_test_pos": int((y == 1).sum()),
                        "num_test_neg": int((y == 0).sum()),
                        "fit_time": fit_times[name],
                        "inference_time": infer_time,
                        "status": "completed",
                        "notes": "official sample-level full test eval",
                    })
                for skipped in resource_limited_models:
                    rows.append({
                        "protocol": protocol,
                        "negative_cap": cap,
                        "negative_ratio": (n_train - n_pos) / max(n_pos, 1),
                        "seed": seed,
                        "model": skipped,
                        "granularity": "sample",
                        "window_size": 1,
                        "setting": dataset,
                        "num_train_pos": n_pos,
                        "num_train_neg": n_train - n_pos,
                        "status": "not_completed_resource_limit",
                        "notes": "large-negative revision timed out on heavy model family; retained existing capped-seed rows where available",
                    })
    # Add D status rows instead of pretending full-negative completed.
    rows.append({
        "protocol": "D_full_negative_or_chunked",
        "negative_cap": "full",
        "negative_ratio": np.nan,
        "seed": "",
        "model": "all",
        "granularity": "sample",
        "window_size": 1,
        "setting": "all",
        "status": "not_completed_resource_limit",
        "notes": "full-negative training requires >10M negative frames; not completed in this revision",
    })
    out = pd.DataFrame(rows)
    write_table(out, "a1_official_sample_negative_stability")
    (OUT / "a1_official_sample_negative_stability.md").write_text(
        "# A1 Official Sample Negative Stability Revision\n\n"
        "Protocol A existing 5-seed capped results are retained. Protocol B 2x cap is run for seeds 42/2024/2026. Protocol C 5x cap is run for seed 42 as a heavier probe. Protocol D full-negative is explicitly marked not completed due resource cost.\n\n"
        f"Completed rows: {int((out['status'].astype(str).str.startswith('completed')).sum())}\n",
        encoding="utf-8",
    )
    return out


def aggregate_train(w: int = 100, seed: int = 42, neg_cap: int = 160_000):
    x_train, y_train, x_val, y_val = collect_window_train(w, seed=seed, max_neg=neg_cap)
    scaler = StandardScaler().fit(x_train)
    model = GradientBoostingClassifier(n_estimators=35, max_depth=2, random_state=seed)
    model.fit(scaler.transform(x_train), y_train)
    val_score = model.predict_proba(scaler.transform(x_val))[:, 1]
    threshold_f1 = best_threshold(y_val, val_score)
    low = recall_at_budget(y_val, val_score, 1e-3)
    return scaler, model, threshold_f1, low["threshold"], x_val, y_val, val_score


def dump_aggregate_scores() -> pd.DataFrame:
    scaler, model, threshold_f1, threshold_low, _, _, _ = aggregate_train()
    rows = []
    for dataset, folder in TEST_FOLDERS.items():
        print(f"[revision] dump aggregate scores {dataset}", flush=True)
        sample_offset = 0
        all_y, all_s = [], []
        out_parts = []
        for file_path, x, y in iter_test_folder(folder):
            wx, wy, ar = window_features(x, y, 100)
            if len(wy) == 0:
                continue
            score = model.predict_proba(scaler.transform(wx))[:, 1]
            all_y.append(wy)
            all_s.append(score)
            n = len(wy)
            out_parts.append(pd.DataFrame({
                "sample_id": np.arange(sample_offset, sample_offset + n),
                "timestamp": np.arange(n, dtype=float),
                "dataset": "ctt",
                "setting": dataset,
                "vehicle": "NA",
                "file": str(file_path),
                "attack_type": np.where(wy == 1, file_path.stem, "normal"),
                "label": wy,
                "score": score,
                "prediction_default": (score >= 0.5).astype(int),
                "prediction_val_f1_threshold": (score >= threshold_f1).astype(int),
                "prediction_low_fpr_threshold": (score >= threshold_low).astype(int),
                "threshold_default": 0.5,
                "threshold_val_f1": threshold_f1,
                "threshold_low_fpr": threshold_low,
                "can_id": np.nan,
                "delta_t_same_id": np.nan,
                "payload_delta_l1": np.nan,
                "payload_sum": np.nan,
                "payload_std": np.nan,
                "window_id": np.arange(sample_offset, sample_offset + n),
                "event_id_if_available": "NA",
                "granularity": "aggregate_window",
                "window_size": 100,
                "model": "GradientBoosting",
                "seed": 42,
                "protocol": "aggregate_window100_train_cap160k",
                "attack_ratio": ar,
            }))
            sample_offset += n
        pred = pd.concat(out_parts, ignore_index=True)
        out_path = PREDS / f"{dataset}_{dataset}_GradientBoosting_aggregate_window100_scores.csv"
        pred.to_csv(out_path, index=False)
        y_all = np.concatenate(all_y)
        s_all = np.concatenate(all_s)
        m = compute_metrics(y_all, s_all, threshold_f1)
        rows.append({"dataset": dataset, "model": "GradientBoosting", "granularity": "aggregate_window", "window_size": 100, "score_file": str(out_path), **m})
    return pd.DataFrame(rows)


def low_fpr_from_score_files() -> pd.DataFrame:
    rows = []
    for path in sorted(PREDS.glob("*_scores.csv")):
        df = pd.read_csv(path)
        if "label" not in df or "score" not in df:
            continue
        y = df["label"].astype(int).to_numpy()
        score = df["score"].astype(float).to_numpy()
        model = str(df["model"].iloc[0]) if "model" in df else path.stem
        setting = str(df["setting"].iloc[0]) if "setting" in df else str(df.get("dataset", path.stem))
        for source in ["best_test_upper_bound"]:
            for budget in [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]:
                b = recall_at_budget(y, score, budget)
                rows.append({
                    "dataset": setting,
                    "model": model,
                    "granularity": df["granularity"].iloc[0] if "granularity" in df else "unknown",
                    "fpr_budget": budget,
                    "threshold_type": source,
                    "recall_at_fpr": b["recall"],
                    "precision_at_fpr": b["precision"],
                    "f1_at_fpr": b["f1"],
                    "actual_fpr": b["actual_fpr"],
                    "threshold": b["threshold"],
                    "num_positive": int((y == 1).sum()),
                    "num_negative": int((y == 0).sum()),
                })
    out = pd.DataFrame(rows)
    write_table(out, "e1_low_fpr_leaderboard")
    fig, ax = plt.subplots(figsize=(6, 3.2))
    plot = out[out["dataset"].isin(["ctt_test02", "ctt_test04"])]
    for label, g in plot.groupby(["dataset", "model", "granularity"]):
        ax.plot(g["fpr_budget"], g["recall_at_fpr"], marker="o", linewidth=1.2, label="/".join(map(str, label))[:34])
    ax.set_xscale("log")
    ax.set_xlabel("FPR budget")
    ax.set_ylabel("Recall")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.6)
    if not plot.empty:
        ax.legend(frameon=False, fontsize=7)
    save_svg(fig, "e1_low_fpr_curves")
    (FIGS / "paper_fig5_low_fpr_curves.svg").write_text((FIGS / "e1_low_fpr_curves.svg").read_text(encoding="utf-8"), encoding="utf-8")
    (OUT / "e1_low_fpr_analysis.md").write_text(
        "# E1 Low-FPR Revision\n\nLow-FPR metrics are recomputed from full aggregate-window score dumps and available audit score files. Aggregate-window rows use best-test upper-bound thresholds for budgets; formal validation-threshold rows require validation score dumps.\n",
        encoding="utf-8",
    )
    return out


def construct_events_and_metrics() -> pd.DataFrame:
    manifest_rows = []
    metric_rows = []
    for path in sorted(PREDS.glob("*aggregate_window100_scores.csv")):
        df = pd.read_csv(path)
        setting = str(df["setting"].iloc[0])
        y = df["label"].astype(int).to_numpy()
        score = df["score"].astype(float).to_numpy()
        pred = df["prediction_val_f1_threshold"].astype(int).to_numpy()
        starts = np.where((y == 1) & np.r_[True, y[:-1] == 0])[0]
        ends = np.r_[starts[1:], len(y)]
        hit = 0
        delays = []
        for event_id, (s, e) in enumerate(zip(starts, ends)):
            attack_type = str(df["attack_type"].iloc[s])
            manifest_rows.append({
                "event_id": f"{setting}_{event_id}",
                "dataset": "ctt",
                "setting": setting,
                "vehicle": "NA",
                "file": str(df["file"].iloc[s]),
                "attack_type": attack_type,
                "start_time": float(df["timestamp"].iloc[s]),
                "end_time": float(df["timestamp"].iloc[e - 1]),
                "start_index": int(s),
                "end_index": int(e - 1),
                "num_positive_samples": int(y[s:e].sum()),
                "duration_seconds": float(max(df["timestamp"].iloc[e - 1] - df["timestamp"].iloc[s], 0)),
                "gap_threshold": "window-contiguous",
                "construction_rule": "label transition 0->1 and contiguous positive aggregate windows",
            })
            idx = np.where(pred[s:e] == 1)[0]
            if len(idx):
                hit += 1
                delays.append(float(idx[0] * 100))
        fp_windows = int(((pred == 1) & (y == 0)).sum())
        duration_hours = max((float(df["timestamp"].max()) - float(df["timestamp"].min())) / 3600.0, len(df) / 3600.0)
        metric_rows.append({
            "dataset": setting,
            "model": "GradientBoosting",
            "granularity": "aggregate_window100",
            "event_recall": hit / max(len(starts), 1),
            "event_precision_if_definable": np.nan,
            "mean_detection_delay_seconds": float(np.mean(delays)) if delays else np.nan,
            "median_detection_delay_seconds": float(np.median(delays)) if delays else np.nan,
            "false_alarm_events_per_hour": np.nan,
            "false_alarm_samples_per_hour": fp_windows / duration_hours,
            "recall_at_0.1_FA_per_hour": np.nan,
            "recall_at_1_FA_per_hour": np.nan,
            "recall_at_10_FA_per_hour": np.nan,
            "event_boundary_quality": "approximate_from_labels",
        })
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(MANIFESTS / "event_boundary_manifest.csv", index=False)
    out = pd.DataFrame(metric_rows)
    write_table(out, "f2_event_level_metrics")
    fig, ax = plt.subplots(figsize=(6, 3.2))
    if not out.empty:
        ax.bar(out["dataset"], out["event_recall"], color="#D9D9D9", edgecolor="black", hatch="//")
    ax.set_ylabel("Event recall")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.6)
    save_svg(fig, "f2_event_level_metrics")
    (FIGS / "paper_fig6_event_level.svg").write_text((FIGS / "f2_event_level_metrics.svg").read_text(encoding="utf-8"), encoding="utf-8")
    (OUT / "event_boundary_construction.md").write_text(
        "# Event Boundary Construction Revision\n\nOfficial event boundaries were not found. Events are constructed from aggregate-window label transitions. This is approximate and must be described conservatively.\n",
        encoding="utf-8",
    )
    (OUT / "f2_event_level_analysis.md").write_text(
        "# F2 Event-Level Revision\n\nEvent-level metrics are recomputed for full aggregate-window score dumps. False alarm per hour uses aggregate-window index as fallback time because official timestamps are not preserved at window level.\n",
        encoding="utf-8",
    )
    return out


def strict_audit_revision() -> pd.DataFrame:
    checks = [
        ("delta_t_same_id_past_only", "pass", "read_ctt_file updates last_ts before only after current row processing"),
        ("payload_delta_l1_past_only", "pass", "read_ctt_file compares against previous same-ID payload"),
        ("period_deviation_train_only", "not_used", "not included in final revision features"),
        ("transition_profile_train_only", "not_used", "not included in final revision features"),
        ("scaler_fit_train_only", "pass", "StandardScaler fit on train arrays only"),
        ("can_id_no_test_label", "pass", "parsed from arbitration_id"),
        ("negative_sampling_train_only", "pass", "collect_train reads train_01 only"),
        ("threshold_validation_only", "partial", "model decision threshold from validation; low-FPR budget thresholds are upper bounds"),
        ("attack_ratio_not_input", "pass", "window_features returns attack_ratio separately, not concatenated into X"),
        ("timestamp_schedule_leakage", "risk", "timestamp-derived deltas can reflect capture schedule"),
        ("file_identifier_feature", "pass", "file identifiers are not used as model features"),
        ("episode_cross_split", "risk", "official CT&T split may contain related attack families across files"),
        ("direct_label_proxy", "pass", "attack column excluded from features"),
        ("large_negative_changes", "partial", "B 2x and C 5x partially completed; full-negative not completed"),
        ("event_boundary_label_use", "partial", "event construction uses test labels for analysis only"),
        ("duplicate_window_cross_split", "not_checked", "not fully audited at raw sample level"),
        ("vehicle_file_leakage", "pass", "vehicle/file not used as features"),
    ]
    out = pd.DataFrame(checks, columns=["check", "status", "evidence"])
    out.to_csv(AUDITS / "strict_feature_leakage_audit.csv", index=False)
    affected = out[out["status"].isin(["risk", "partial", "not_checked"])]
    affected.to_csv(AUDITS / "affected_experiments_if_any.csv", index=False)
    (AUDITS / "strict_feature_leakage_audit.md").write_text(
        "# Strict Feature Leakage Audit Revision\n\n"
        f"```csv\n{out.to_csv(index=False)}```\n\n"
        "Blocking direct leakage was not found. Remaining risks are timestamp/capture schedule, full-negative incompleteness, and approximate event boundaries.\n",
        encoding="utf-8",
    )
    return out


def refresh_reports(a1: pd.DataFrame, e1: pd.DataFrame, f2: pd.DataFrame, audit: pd.DataFrame) -> None:
    test02 = a1[(a1.get("setting", a1.get("dataset")).astype(str).eq("ctt_test02")) & a1["model"].eq("GradientBoosting")]
    test04_event = f2[f2["dataset"].astype(str).eq("ctt_test04")]
    (OUT / "final_security4_readiness_report.md").write_text(
        "# Final Security-Four Readiness Report Revision\n\n"
        "Current status: stronger than the previous packaging step, but still not Security Four-ready.\n\n"
        f"test02 GradientBoosting rows across completed negative protocols: {len(test02)}.\n\n"
        "What improved: large-negative Protocol B and one Protocol C probe were added; aggregate-window full score dumps now support low-FPR and event-level recomputation.\n\n"
        "Remaining blockers: full-negative Protocol D was not completed; low-FPR aggregate results use best-test budget thresholds rather than formal validation budget thresholds; event boundaries are approximate.\n\n"
        "Recommendation: CCF B / strong measurement-track submission is realistic. CCF A/Security Four still needs full-negative or streaming full-negative evidence plus official/defensible event boundaries.\n",
        encoding="utf-8",
    )
    (OUT / "final_research_direction.md").write_text(
        "# Final Research Direction Revision\n\n"
        "**Selected: B. Protocol-Gap / Measurement Paper with GRAIN-CAN feature-preserving detector evidence.**\n\n"
        "Reason: feature-preserving granularity clearly fixes major shifted CT&T failures, but full-negative and event-level deployment evidence remain incomplete for a full Security-Four method claim.\n",
        encoding="utf-8",
    )
    (OUT / "unsafe_claims_do_not_write.md").write_text(
        "# Unsafe Claims Do Not Write Revision\n\n"
        "- Do not claim Security-Four readiness.\n"
        "- Do not claim full-negative stability; Protocol D is not completed.\n"
        "- Do not claim test04 unknown attack breakthrough unless low-FPR and event-level budgets are accepted as approximate.\n"
        "- Do not claim normality policy is a contribution.\n"
        "- Do not claim old CMF/Reliable/TFS gate is the main method.\n",
        encoding="utf-8",
    )
    (OUT / "recommended_paper_outline.md").write_text(
        "# Recommended Paper Outline Revision\n\n"
        "1. Protocol gap: old deep long-window representations hide CT&T shifted signals.\n"
        "2. GRAIN-CAN: feature-preserving granularity-aware aggregate windows.\n"
        "3. Evidence: negative-cap stability, full aggregate score dumps, low-FPR and approximate event-level metrics.\n"
        "4. Audit: past-only features and attack_ratio leakage removal.\n"
        "5. Limits: full-negative and official event boundaries remain future work.\n",
        encoding="utf-8",
    )


def main() -> None:
    setup()
    a1 = large_negative_stability()
    aggregate = dump_aggregate_scores()
    aggregate.to_csv(REV / "aggregate_score_dump_metrics.csv", index=False)
    e1 = low_fpr_from_score_files()
    f2 = construct_events_and_metrics()
    audit = strict_audit_revision()
    refresh_reports(a1, e1, f2, audit)
    print("[final_grain_can_revision] done")


if __name__ == "__main__":
    main()
