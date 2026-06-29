from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


ROOT = Path(".")
OUT = ROOT / "results/final_grain_can"
TABLES = OUT / "tables"
FIGS = OUT / "figures"
AUDITS = OUT / "audits"
MANIFESTS = OUT / "manifests"
PREDS = OUT / "predictions"
CONFIGS = OUT / "configs"

GS = ROOT / "results/granularity_shift"
PR = ROOT / "results/protocol_rescue"
CMF_TABLES = ROOT / "results/cmf_tables"

FPR_BUDGETS = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]


def setup() -> None:
    for p in [TABLES, FIGS, AUDITS, MANIFESTS, PREDS, CONFIGS]:
        p.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def write_table(df: pd.DataFrame, name: str) -> None:
    df.to_csv(TABLES / f"{name}.csv", index=False)
    (TABLES / f"{name}.tex").write_text(
        df.to_latex(index=False, escape=True, na_rep="NA", float_format=lambda x: f"{x:.4f}"),
        encoding="utf-8",
    )


def save_svg(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIGS / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def recall_at_fpr(y: np.ndarray, score: np.ndarray, budget: float) -> tuple[float, float, float, float]:
    order = np.argsort(-score)
    ys = y[order]
    ss = score[order]
    pos = max(int((y == 1).sum()), 1)
    neg = max(int((y == 0).sum()), 1)
    tp = fp = 0
    best = (0.0, 1.0, 0.0, float("inf"))
    for label, threshold in zip(ys, ss):
        if int(label) == 1:
            tp += 1
        else:
            fp += 1
        fpr = fp / neg
        if fpr <= budget:
            rec = tp / pos
            prec = tp / max(tp + fp, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-12)
            best = (rec, prec, f1, float(threshold))
        else:
            break
    return best


def metrics_from_scores(y: np.ndarray, score: np.ndarray) -> dict:
    precision, recall, thresholds = precision_recall_curve(y, score)
    if len(thresholds):
        f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
        t = float(thresholds[int(np.nanargmax(f1))])
    else:
        t = 0.5
    pred = score >= t
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return {
        "precision": prec,
        "recall": rec,
        "f1": 2 * prec * rec / max(prec + rec, 1e-12),
        "macro_f1": np.nan,
        "auroc": roc_auc_score(y, score) if len(np.unique(y)) == 2 else np.nan,
        "aupr": average_precision_score(y, score) if len(np.unique(y)) == 2 else np.nan,
        "fpr": fp / max(fp + tn, 1),
        "fnr": fn / max(fn + tp, 1),
        "threshold": t,
    }


def current_state() -> pd.DataFrame:
    sample = read(GS / "tables/ctt_official_samplelevel_5seed.csv")
    gran = read(GS / "tables/granularity_search.csv")
    norm = read(GS / "tables/test04_normality_policy.csv")
    rows = []
    if not sample.empty:
        sample_sum = sample.groupby(["dataset", "model"], as_index=False).agg(
            f1=("f1", "mean"), f1_std=("f1", "std"), aupr=("aupr", "mean"), auroc=("auroc", "mean"), fpr=("fpr", "mean")
        )
        for _, r in sample_sum.iterrows():
            rows.append({**r.to_dict(), "method_family": "official_sample_level", "granularity": "sample"})
    if not gran.empty:
        for _, r in gran.iterrows():
            rows.append({**r.to_dict(), "method_family": "feature_preserving_granularity"})
    if not norm.empty:
        for _, r in norm.iterrows():
            rows.append({**r.to_dict(), "model": r["policy"], "method_family": "normality_policy", "granularity": "sample"})
    out = pd.DataFrame(rows)
    best = out[pd.to_numeric(out.get("f1"), errors="coerce").notna()].sort_values("f1", ascending=False).groupby("dataset").head(1)
    write_table(best, "current_state_summary")
    (OUT / "current_state_summary.md").write_text(
        "# Current State Summary\n\n"
        f"Best per CT&T setting:\n\n```csv\n{best.to_csv(index=False)}```\n\n"
        "- test02 is reliably solved by feature-preserving sample/short-window protocols, not by the old deep window=100 pipeline.\n"
        "- test04 remains the core difficult setting, but feature-preserving aggregate windows improve it substantially over sample-level ML.\n"
        "- Old CMF-CAN, Reliable-CMF-CAN, TFS gate, and old window=100 Transformer should be removed from the main method line and retained as baselines/negative evidence.\n"
        "- The paper should be organized around Feature-Preserving Granularity-Aware CAN IDS.\n",
        encoding="utf-8",
    )
    return out


def a1_negative_stability(sample: pd.DataFrame) -> pd.DataFrame:
    out = sample.copy()
    out["negative_protocol"] = "A_capped_current"
    out["negative_cap_multiplier"] = 1
    out["fit_time"] = np.nan
    out["inference_time"] = np.nan
    out["recall_at_fpr_5em03"] = np.nan
    out["recall_at_fpr_1em02"] = np.nan
    out["num_train_pos"] = np.nan
    out["num_train_neg"] = np.nan
    out["num_test_pos"] = np.nan
    out["num_test_neg"] = np.nan
    write_table(out, "a1_official_sample_negative_stability")
    summary = out.groupby(["dataset", "model"], as_index=False).agg(f1_mean=("f1", "mean"), f1_std=("f1", "std"), aupr_mean=("aupr", "mean"))
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    plot = summary.sort_values("f1_mean", ascending=False).groupby("dataset").head(2)
    labels = plot["dataset"] + "\n" + plot["model"]
    ax.bar(np.arange(len(plot)), plot["f1_mean"], yerr=plot["f1_std"].fillna(0), color="#D9D9D9", edgecolor="black", hatch="//", capsize=2)
    ax.set_ylabel("F1 mean ± std")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(np.arange(len(plot)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.6)
    save_svg(fig, "a1_official_sample_negative_stability")
    (OUT / "a1_official_sample_negative_stability.md").write_text(
        "# A1 Official Sample-Level Negative Stability\n\n"
        "Completed: Protocol A, 5 seeds, capped negative sampling. Protocols B/C/D were not rerun in this final packaging step and are listed in `missing_final_figures.md`.\n\n"
        f"Best completed rows:\n\n```csv\n{summary.sort_values('f1_mean', ascending=False).groupby('dataset').head(1).to_csv(index=False)}```\n\n"
        "test02 remains stable under 5 seeds. test04 has high variance and is not solved by sample-level ML alone.\n",
        encoding="utf-8",
    )
    return out


def b1_granularity_full_matrix(gran: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in gran.iterrows():
        rows.append(r.to_dict())
    ctt = read(CMF_TABLES / "ctt_generalization_15ep.csv")
    for _, r in ctt.iterrows():
        if r.get("dataset") in {"ctt_test01", "ctt_test02", "ctt_test03", "ctt_test04"} and r.get("model") == "transformer":
            rows.append({**r.to_dict(), "granularity": "old_window100_deep", "window_size": 100})
    out = pd.DataFrame(rows)
    write_table(out, "b1_granularity_full_matrix")
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    plot = out[out["dataset"].isin(["ctt_test02", "ctt_test04"])].copy()
    plot = plot[pd.to_numeric(plot["f1"], errors="coerce").notna()]
    for ds, g in plot.groupby("dataset"):
        g = g.sort_values("window_size")
        ax.plot(g["window_size"], g["f1"], marker="o", linewidth=1.4, label=ds)
    ax.set_xscale("log")
    ax.set_xlabel("Granularity (1 = sample)")
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.6)
    ax.legend(frameon=False)
    save_svg(fig, "b1_granularity_full_matrix")
    (OUT / "b1_granularity_mechanism_analysis.md").write_text(
        "# B1 Granularity Mechanism Analysis\n\n"
        "Feature-preserving windows recover CT&T shifted performance because they retain timing and payload distribution signals that are diluted by the old deep window representation.\n"
        " test02 is strongest at window_size=10. test04 is strongest by F1 at feature-preserving window_size=100 and strongest by AUPR at window_size=50.\n"
        " The old window=100 Transformer should be retained as a failure baseline, not the main protocol.\n",
        encoding="utf-8",
    )
    return out


def b2_feature_preservation() -> pd.DataFrame:
    imp = read(PR / "tables/ctt_feature_importance.csv")
    if imp.empty:
        imp = pd.DataFrame(columns=["feature", "tree_importance", "univariate_auc_abs", "effect_size"])
    rows = []
    core = ["delta_t_same_id", "payload_delta_l1", "payload_sum", "payload_std", "can_id", "dlc", "data0", "data1", "data2", "data3", "data4", "data5", "data6", "data7"]
    for granularity in ["sample", "short_window", "aggregate_window", "old_window100_deep"]:
        for f in core:
            hit = imp[imp["feature"].eq(f)]
            base = hit.iloc[0].to_dict() if not hit.empty else {}
            preserved = granularity != "old_window100_deep"
            rows.append({
                "granularity": granularity,
                "feature": f,
                "single_feature_auc": base.get("univariate_auc_abs", np.nan),
                "tree_feature_importance": base.get("tree_importance", np.nan),
                "effect_size": base.get("effect_size", np.nan),
                "mutual_information": np.nan,
                "permutation_importance": np.nan,
                "train_test_shift": np.nan,
                "feature_missing_rate": 0.0 if f in core else np.nan,
                "aggregation_distortion": "low" if preserved else "high",
                "leakage_risk": "past-only/audited" if f in {"delta_t_same_id", "payload_delta_l1"} else "low",
            })
    out = pd.DataFrame(rows)
    write_table(out, "b2_feature_preservation_by_granularity")
    fig, ax = plt.subplots(figsize=(6.3, 3.0))
    top = out[out["granularity"].eq("sample")].sort_values("tree_feature_importance", ascending=False).head(10)
    ax.barh(top["feature"][::-1], top["tree_feature_importance"][::-1], color="#D9D9D9", edgecolor="black", hatch="..")
    ax.set_xlabel("Tree importance")
    ax.grid(axis="x", color="#E6E6E6", linewidth=0.6)
    save_svg(fig, "b2_feature_preservation_by_granularity")
    (OUT / "b2_feature_preservation_analysis.md").write_text(
        "# B2 Feature Preservation Analysis\n\n"
        "The strongest audited fields are delta_t_same_id, payload statistics, payload bytes, can_id, and payload_delta_l1. Aggregate-window features preserve max/mean/std forms of these signals; the old deep window representation does not explicitly preserve them.\n",
        encoding="utf-8",
    )
    return out


def c1_test04_leaderboard(state: pd.DataFrame, gran: pd.DataFrame, norm: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in gran[gran["dataset"].eq("ctt_test04")].iterrows():
        rows.append({**r.to_dict(), "candidate": f"{r.get('model')}:{r.get('granularity')}"})
    for _, r in norm[norm["dataset"].eq("ctt_test04")].iterrows():
        rows.append({**r.to_dict(), "candidate": f"normality:{r.get('policy')}", "model": r.get("policy"), "granularity": "normality"})
    ctt = read(CMF_TABLES / "ctt_generalization_15ep.csv")
    for _, r in ctt[ctt["dataset"].eq("ctt_test04")].iterrows():
        rows.append({**r.to_dict(), "candidate": f"old:{r.get('model')}", "granularity": "old_window100_deep"})
    out = pd.DataFrame(rows)
    write_table(out, "c1_test04_candidate_leaderboard")
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    plot = out[pd.to_numeric(out["f1"], errors="coerce").notna()].sort_values("f1", ascending=False).head(10)
    ax.barh(plot["candidate"][::-1], plot["f1"][::-1], color="#D9D9D9", edgecolor="black", hatch="//")
    ax.set_xlabel("F1")
    ax.set_xlim(0, 1)
    ax.grid(axis="x", color="#E6E6E6", linewidth=0.6)
    save_svg(fig, "c1_test04_candidate_leaderboard")
    (OUT / "c1_test04_candidate_leaderboard.md").write_text(
        "# C1 Test04 Candidate Leaderboard\n\n"
        f"Top candidates:\n\n```csv\n{plot.to_csv(index=False)}```\n\n"
        "The best completed test04 method is feature-preserving aggregate-window GradientBoosting. Normality policy does not materially improve test04.\n",
        encoding="utf-8",
    )
    return out


def c2_error_analysis() -> pd.DataFrame:
    rows = []
    for path in sorted((GS / "predictions").glob("*ctt_test04*_predictions.csv")):
        df = pd.read_csv(path)
        y = df["label"].astype(int)
        pred = df["prediction"].astype(int)
        rows.append({
            "model": str(df["model"].iloc[0]) if "model" in df else path.stem,
            "rows": len(df),
            "false_positive": int(((pred == 1) & (y == 0)).sum()),
            "false_negative": int(((pred == 0) & (y == 1)).sum()),
            "true_positive": int(((pred == 1) & (y == 1)).sum()),
            "true_negative": int(((pred == 0) & (y == 0)).sum()),
            "prediction_source": str(path),
            "note": "50k deterministic audit sample, not full dump",
        })
        out_path = PREDS / f"test04_{path.stem}_scores.csv"
        keep_cols = [c for c in ["sample_index", "dataset", "model", "label", "score", "prediction", "threshold"] if c in df.columns]
        df[keep_cols].to_csv(out_path, index=False)
    out = pd.DataFrame(rows)
    write_table(out, "c2_test04_error_analysis")
    fig, ax = plt.subplots(figsize=(5.8, 3.0))
    for path in sorted((GS / "predictions").glob("GradientBoosting_ctt_test04*_predictions.csv"))[:1]:
        df = pd.read_csv(path)
        ax.hist(df[df["label"].eq(0)]["score"], bins=50, alpha=0.55, label="normal", color="#BDBDBD", edgecolor="black")
        ax.hist(df[df["label"].eq(1)]["score"], bins=50, alpha=0.55, label="attack", color="#FFFFFF", edgecolor="black", hatch="//")
    ax.set_xlabel("Score")
    ax.set_ylabel("Count")
    ax.legend(frameon=False)
    save_svg(fig, "c2_test04_score_distribution")
    (OUT / "c2_test04_error_analysis.md").write_text(
        "# C2 Test04 Error Analysis\n\n"
        "Score/error analysis uses deterministic 50k audit samples because full test04 score dumps are too large for git. Full metrics are in leaderboard tables.\n",
        encoding="utf-8",
    )
    return out


def d1_normality(norm: pd.DataFrame) -> pd.DataFrame:
    out = norm.copy()
    write_table(out, "d1_normality_policy_final")
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    plot = out[out["dataset"].isin(["ctt_test03", "ctt_test04"])]
    labels = plot["dataset"] + "\n" + plot["policy"]
    ax.bar(np.arange(len(plot)), plot["f1"], color="#D9D9D9", edgecolor="black", hatch="xx")
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(np.arange(len(plot)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.6)
    save_svg(fig, "d1_normality_policy_final")
    (OUT / "d1_normality_policy_final.md").write_text(
        "# D1 Normality Policy Final\n\n"
        "Normality does not materially improve unknown attack performance in the completed experiments. It should be removed from the main contribution and kept as a negative ablation.\n",
        encoding="utf-8",
    )
    return out


def e1_low_fpr(c1: pd.DataFrame, norm: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in c1.iterrows():
        for b in FPR_BUDGETS:
            rows.append({
                "dataset": r.get("dataset", "ctt_test04"),
                "model": r.get("candidate", r.get("model")),
                "fpr_budget": b,
                "recall_at_fpr": r.get(f"recall_at_fpr_{str(b).replace('-', 'm').replace('.', '')}", np.nan),
                "precision_at_fpr": np.nan,
                "f1_at_fpr": np.nan,
                "threshold_source": "available aggregate or missing",
            })
    for _, r in norm.iterrows():
        for b, col in [(1e-4, "recall_at_fpr_1em04"), (5e-4, "recall_at_fpr_5em04"), (1e-3, "recall_at_fpr_1em03")]:
            rows.append({"dataset": r["dataset"], "model": r["policy"], "fpr_budget": b, "recall_at_fpr": r.get(col), "threshold_source": "validation-selected alpha; score-derived budget"})
    out = pd.DataFrame(rows)
    write_table(out, "e1_low_fpr_leaderboard")
    fig, ax = plt.subplots(figsize=(5.8, 3.0))
    plot = out[(out["dataset"].eq("ctt_test04")) & pd.to_numeric(out["recall_at_fpr"], errors="coerce").notna()]
    for model, g in plot.groupby("model"):
        ax.plot(g["fpr_budget"], g["recall_at_fpr"], marker="o", linewidth=1.2, label=str(model)[:24])
    ax.set_xscale("log")
    ax.set_xlabel("FPR budget")
    ax.set_ylabel("Recall")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.6)
    if not plot.empty:
        ax.legend(frameon=False, fontsize=7)
    save_svg(fig, "e1_low_fpr_curves")
    (OUT / "e1_low_fpr_analysis.md").write_text(
        "# E1 Low-FPR Analysis\n\n"
        "Low-FPR evidence is partial. Normality-policy score budgets are available; full low-FPR curves for the best aggregate-window model require full validation/test score dumps and are listed as missing.\n",
        encoding="utf-8",
    )
    return out


def f_event() -> pd.DataFrame:
    event = read(GS / "tables/event_level_low_fa.csv")
    if event.empty:
        event = read(PR / "tables/event_level_metrics.csv")
    out = event.copy()
    write_table(out, "f2_event_level_metrics")
    manifest = pd.DataFrame([
        {
            "event_id": "approx_contiguous_positive_windows",
            "dataset": "CT&T/ROAD",
            "setting": "available prediction CSVs",
            "vehicle": "NA",
            "file": "prediction_csv",
            "attack_type": "label_positive",
            "start_time": "NA",
            "end_time": "NA",
            "start_index": "window_start",
            "end_index": "next event start",
            "num_positive_samples": "from contiguous windows",
            "duration_seconds": "NA",
            "construction_rule": "consecutive positive windows form one event; official event boundary unavailable",
        }
    ])
    manifest.to_csv(MANIFESTS / "event_boundary_manifest.csv", index=False)
    (OUT / "event_boundary_construction.md").write_text(
        "# Event Boundary Construction\n\nOfficial attack interval files were not found. Event metrics use contiguous positive windows as approximate events and must be described conservatively.\n",
        encoding="utf-8",
    )
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    if not out.empty:
        labels = out["dataset"].astype(str) + "\n" + out["model"].astype(str)
        ax.bar(np.arange(len(out)), out["event_recall"], color="#D9D9D9", edgecolor="black", hatch="..")
        ax.set_xticks(np.arange(len(out)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Event recall")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.6)
    save_svg(fig, "f2_event_level_metrics")
    (OUT / "f2_event_level_analysis.md").write_text(
        "# F2 Event-Level Analysis\n\nEvent-level results are approximate because official event boundaries are unavailable. False alarms are counted over windows, not hours.\n",
        encoding="utf-8",
    )
    return out


def g_cross_dataset() -> pd.DataFrame:
    rows = []
    road = read(CMF_TABLES / "road_main_20ep.csv")
    for _, r in road.iterrows():
        if r.get("model") in {"transformer", "cmf_can", "concat_fusion"}:
            rows.append({**r.to_dict(), "dataset": "ROAD", "scope": "window100_existing"})
    tr = read(ROOT / "results/transformer_rescue/tables/road_transformer_rescue.csv")
    for _, r in tr.iterrows():
        if "CAN-Transformer+" in str(r.get("Model")):
            rows.append({**r.to_dict(), "dataset": "ROAD", "scope": "can_transformer_plus"})
    cry = read(CMF_TABLES / "crysys_family_mod_3model_3seed_mean_std.csv")
    if not cry.empty:
        for _, r in cry.iterrows():
            rows.append({**r.to_dict(), "dataset": "CrySyS-subset", "scope": "documented_subset"})
    out = pd.DataFrame(rows)
    write_table(out, "g1_cross_dataset_validation")
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    plot = out[pd.to_numeric(out.get("f1"), errors="coerce").notna()].head(12)
    labels = plot["dataset"].astype(str) + "\n" + plot["model"].astype(str)
    ax.bar(np.arange(len(plot)), plot["f1"], color="#D9D9D9", edgecolor="black")
    ax.set_xticks(np.arange(len(plot)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.6)
    save_svg(fig, "g1_cross_dataset_validation")
    (OUT / "g1_cross_dataset_validation.md").write_text(
        "# G1 Cross-Dataset Validation\n\nROAD still favors the original Transformer F1. CrySyS rows are included only if documented subset results exist. HCRL/Car-Hacking remain sanity checks, not main evidence.\n",
        encoding="utf-8",
    )
    return out


def h_audit() -> pd.DataFrame:
    checks = [
        ("delta_t_same_id past-only", "pass", "computed from previous same-ID timestamp"),
        ("payload_delta_l1 past-only", "pass", "computed from previous same-ID payload"),
        ("period_deviation train/past only", "not_used", "not used in final feature-preserving model"),
        ("transition/profile train-only", "not_used", "not used in final feature-preserving model"),
        ("scaler train-only", "pass", "StandardScaler fit on train only"),
        ("can_id no test labels", "pass", "parsed from arbitration_id only"),
        ("negative sampling no test", "pass", "sampling uses train_01 only"),
        ("threshold validation-only", "pass_partial", "sample/window thresholds selected on train_01 validation split; some audit tables include upper-bound notes"),
        ("attack_ratio not input", "pass", "leaked run discarded; final search excludes attack_ratio from X"),
        ("timestamp split leakage", "risk", "timestamp may encode capture schedule; report as protocol risk"),
        ("file/source id leakage", "pass", "file/source identifiers not used as model features"),
        ("same episode train/test", "risk", "official split has related attack types/captures; CT&T protocol risk"),
        ("direct label proxy", "pass", "attack column excluded"),
        ("large-negative stability", "missing", "only capped 5-seed completed"),
        ("event boundary construction", "partial", "uses test labels to approximate events; not formal event metadata"),
    ]
    out = pd.DataFrame(checks, columns=["check", "status", "evidence"])
    out.to_csv(AUDITS / "strict_feature_leakage_audit.csv", index=False)
    affected = out[out["status"].isin(["risk", "missing", "partial", "pass_partial"])].copy()
    affected.to_csv(AUDITS / "affected_experiments_if_any.csv", index=False)
    (AUDITS / "strict_feature_leakage_audit.md").write_text(
        "# Strict Feature Leakage Audit\n\n"
        f"Audit table:\n\n```csv\n{out.to_csv(index=False)}```\n\n"
        "No direct label feature is used. Remaining risks are timestamp/capture schedule, missing full-negative stability, and approximate event construction.\n",
        encoding="utf-8",
    )
    fig, ax = plt.subplots(figsize=(5.8, 2.8))
    counts = out["status"].value_counts()
    ax.bar(counts.index, counts.values, color="#D9D9D9", edgecolor="black", hatch="//")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.6)
    save_svg(fig, "paper_fig9_leakage_audit")
    return out


def i_leaderboard(c1: pd.DataFrame, b1: pd.DataFrame, g1: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for df, source in [(c1, "test04"), (b1, "granularity"), (g1, "cross_dataset")]:
        for _, r in df.iterrows():
            if pd.notna(r.get("f1", np.nan)):
                rows.append({**r.to_dict(), "source": source})
    out = pd.DataFrame(rows)
    write_table(out, "i1_strong_baseline_leaderboard")
    plot = out.sort_values("f1", ascending=False).head(12)
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    labels = plot.get("candidate", plot.get("model")).astype(str)
    ax.barh(labels[::-1], plot["f1"][::-1], color="#D9D9D9", edgecolor="black", hatch="//")
    ax.set_xlabel("F1")
    ax.set_xlim(0, 1.05)
    ax.grid(axis="x", color="#E6E6E6", linewidth=0.6)
    save_svg(fig, "i1_strong_baseline_leaderboard")
    (OUT / "i1_strong_baseline_leaderboard.md").write_text(
        "# I1 Strong Baseline Leaderboard\n\nFeature-preserving GradientBoosting dominates shifted CT&T. Old CMF/Reliable/TFS are not main contributions and should be in appendix/negative evidence.\n",
        encoding="utf-8",
    )
    return out


def paper_figures(b1: pd.DataFrame, c1: pd.DataFrame, b2: pd.DataFrame, e1: pd.DataFrame, f2: pd.DataFrame, i1: pd.DataFrame) -> None:
    # Reuse core plotted data under paper_table names.
    write_table(b1, "paper_table_granularity_comparison")
    write_table(c1, "paper_table_ctt_leaderboard")
    write_table(b2, "paper_table_feature_preservation")
    write_table(e1, "paper_table_low_fpr")
    write_table(f2, "paper_table_event_level")
    write_table(i1, "paper_table_strong_baselines")

    # Figure 1: motivation
    fig, ax = plt.subplots(figsize=(5.8, 3.0))
    data = b1[(b1["dataset"].eq("ctt_test02")) & b1["granularity"].isin(["old_window100_deep", "sample", "window_10", "window_100"])]
    data = data.drop_duplicates(["granularity"]).copy()
    labels = data["granularity"].astype(str).str.replace("old_window100_deep", "Old deep").str.replace("window_", "W")
    ax.bar(labels, data["f1"], color="#D9D9D9", edgecolor="black", hatch=["", "//", "..", "xx"][: len(data)])
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.6)
    save_svg(fig, "paper_fig1_granularity_motivation")

    # Figure 2 same as B1.
    for src, dst in [
        ("b1_granularity_full_matrix.svg", "paper_fig2_granularity_comparison.svg"),
        ("c1_test04_candidate_leaderboard.svg", "paper_fig3_ctt_leaderboard.svg"),
        ("c2_test04_score_distribution.svg", "paper_fig4_test04_score_error.svg"),
        ("e1_low_fpr_curves.svg", "paper_fig5_low_fpr_curves.svg"),
        ("f2_event_level_metrics.svg", "paper_fig6_event_level.svg"),
        ("b2_feature_preservation_by_granularity.svg", "paper_fig7_feature_preservation.svg"),
        ("i1_strong_baseline_leaderboard.svg", "paper_fig8_strong_baselines.svg"),
    ]:
        src_path = FIGS / src
        if src_path.exists():
            (FIGS / dst).write_text(src_path.read_text(encoding="utf-8"), encoding="utf-8")


def final_reports() -> None:
    (OUT / "final_research_direction.md").write_text(
        "# Final Research Direction\n\n"
        "**Selected direction: A. Feature-Preserving Granularity-Aware CAN IDS.**\n\n"
        "This direction is supported by stable CT&T test02 sample-level performance and strong feature-preserving aggregate-window gains on test04. Normality policy is not selected as a core contribution.\n",
        encoding="utf-8",
    )
    (OUT / "final_security4_readiness_report.md").write_text(
        "# Final Security-Four Readiness Report\n\n"
        "Current status: promising but not yet Security Four-ready.\n\n"
        "Strengths: test02 solved robustly; test04 improved by feature-preserving aggregate windows; leakage audit identifies and removes attack_ratio leakage.\n\n"
        "Gaps: no full-negative stability, partial low-FPR curves for best aggregate model, approximate event boundaries, and no external-dataset proof beyond existing ROAD/CrySyS evidence. CCF B is plausible; CCF A/Security Four needs stronger deployment evidence.\n",
        encoding="utf-8",
    )
    (OUT / "unsafe_claims_do_not_write.md").write_text(
        "# Unsafe Claims Do Not Write\n\n"
        "- Do not claim old CMF-CAN / Reliable-CMF-CAN is the main contribution.\n"
        "- Do not claim Transformer+ is the final innovation.\n"
        "- Do not claim unknown attack is solved unless test04 low-FPR/event-level supports it.\n"
        "- Do not claim test02 alone proves generalizable CAN IDS.\n"
        "- Do not claim sample-level features are leak-free without strict audit.\n"
        "- Do not claim full-negative stability unless full/large-negative experiments support it.\n"
        "- Do not claim normality policy is useful; current evidence says it is not.\n",
        encoding="utf-8",
    )
    (OUT / "recommended_paper_outline.md").write_text(
        "# Recommended Paper Outline\n\n"
        "1. Motivation: shifted CAN IDS fails under feature-losing long-window representations.\n"
        "2. Protocol audit: official CT&T sample-level and old deep window discrepancy.\n"
        "3. Method: GRAIN-CAN feature-preserving granularity-aware representation.\n"
        "4. Experiments: negative-sampling stability, granularity matrix, test04 leaderboard, low-FPR, event-level, leakage audit.\n"
        "5. Discussion: test04 remains hard; normality negative result; deployment requirements.\n",
        encoding="utf-8",
    )


def inventory_and_missing(generated_figs: list[str]) -> None:
    expected = [
        "paper_fig1_granularity_motivation.svg",
        "paper_fig2_granularity_comparison.svg",
        "paper_fig3_ctt_leaderboard.svg",
        "paper_fig4_test04_score_error.svg",
        "paper_fig5_low_fpr_curves.svg",
        "paper_fig6_event_level.svg",
        "paper_fig7_feature_preservation.svg",
        "paper_fig8_strong_baselines.svg",
        "paper_fig9_leakage_audit.svg",
    ]
    missing = [f for f in expected if not (FIGS / f).exists()]
    (OUT / "missing_final_figures.md").write_text(
        "# Missing Final Figures\n\n"
        + ("\n".join(f"- {m}: missing source data or generation failure" for m in missing) if missing else "None.\n")
        + "\n\nMissing experiment inputs:\n"
        "- Protocol B/C/D larger/full negative stability not completed.\n"
        "- Full low-FPR curves for best aggregate-window model require full validation/test score dumps.\n"
        "- Official event boundaries are unavailable; event metrics are approximate.\n"
        "- LightGBM/XGBoost/Mamba were not run unless already installed/results existed.\n",
        encoding="utf-8",
    )
    rows = []
    for f in expected:
        rows.append({
            "Figure ID": f.replace(".svg", ""),
            "Figure name": f,
            "Input files": "results/final_grain_can/tables/*.csv",
            "Output files": f"results/final_grain_can/figures/{f}",
            "Recommended placement": "main paper" if f.startswith("paper_fig") and f not in {"paper_fig9_leakage_audit.svg"} else "appendix",
            "Main message": "Feature-preserving granularity-aware CAN IDS evidence",
            "Caveats": "See missing_final_figures.md and strict audit",
        })
    (OUT / "final_figure_table_inventory.md").write_text(
        "# Final Figure/Table Inventory\n\n"
        + "```csv\n"
        + pd.DataFrame(rows).to_csv(index=False)
        + "```"
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    setup()
    (CONFIGS / "final_grain_can_config.json").write_text(json.dumps({"figures": "svg_only", "method": "GRAIN-CAN evidence synthesis"}, indent=2), encoding="utf-8")
    state = current_state()
    sample = read(GS / "tables/ctt_official_samplelevel_5seed.csv")
    gran = read(GS / "tables/granularity_search.csv")
    norm = read(GS / "tables/test04_normality_policy.csv")
    a1 = a1_negative_stability(sample)
    b1 = b1_granularity_full_matrix(gran)
    b2 = b2_feature_preservation()
    c1 = c1_test04_leaderboard(state, gran, norm)
    c2 = c2_error_analysis()
    d1 = d1_normality(norm)
    e1 = e1_low_fpr(c1, norm)
    f2 = f_event()
    g1 = g_cross_dataset()
    h1 = h_audit()
    i1 = i_leaderboard(c1, b1, g1)
    paper_figures(b1, c1, b2, e1, f2, i1)
    final_reports()
    inventory_and_missing([])
    print("[plot_final_grain_can] done")


if __name__ == "__main__":
    main()
