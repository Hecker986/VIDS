from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cmf_can.analysis.attack_centric_eval import compute_binary_metrics, compute_ranking_inversion


OUT = Path("results/attack_centric_final")
TABLES = OUT / "tables"
FIGS = OUT / "figures"
LOGS = OUT / "logs"
PRED = OUT / "predictions"
AUDITS = OUT / "audits"
MANIFESTS = OUT / "manifests"
TOOLKIT = OUT / "toolkit"
ASSETS = OUT / "paper_assets"

SOURCES = {
    "dataset_summary": Path("results/cmf_tables/paper_table_dataset_summary_refined.csv"),
    "ctt_grain": Path("results/final_grain_can/tables/b1_granularity_full_matrix.csv"),
    "sample_ml": Path("results/final_grain_can/tables/a1_official_sample_negative_stability.csv"),
    "table13_targets": Path("results/metric_correction_paper/tables/table13_public_targets_full.csv"),
    "table13_metrics": Path("results/metric_correction_paper/tables/full_attack_normal_weighted_metrics.csv"),
    "table13_match": Path("results/metric_correction_paper/tables/table13_metric_hypothesis_matrix.csv"),
    "trivial_test04": Path("results/metric_correction_paper/tables/trivial_baseline_imbalance.csv"),
    "low_fpr": Path("results/final_grain_can/tables/e1_low_fpr_leaderboard.csv"),
    "event": Path("results/final_grain_can/tables/f2_event_level_metrics.csv"),
    "feature": Path("results/final_grain_can/tables/b2_feature_preservation_by_granularity.csv"),
    "external": Path("results/final_grain_can/tables/g1_cross_dataset_validation.csv"),
    "main_results": Path("results/cmf_tables/paper_table_overall_main_results_refined.csv"),
    "metric_correction_benchmark": Path("results/metric_correction_paper/tables/corrected_test04_benchmark_final.csv"),
}

CTT_SETTINGS = {"ctt_test01", "ctt_test02", "ctt_test03", "ctt_test04"}
PLOT_COLORS = ["#1B9E77", "#D95F02", "#7570B3", "#E7298A", "#66A61E", "#E6AB02", "#A6761D", "#1F78B4"]


def setup() -> None:
    for p in [TABLES, FIGS, LOGS, PRED, AUDITS, MANIFESTS, TOOLKIT, ASSETS]:
        p.mkdir(parents=True, exist_ok=True)
    inventory_roots = [
        "results/metric_correction_paper",
        "results/table13_metric_forensics",
        "results/final_grain_can",
        "results/granularity_shift",
        "results/protocol_rescue",
        "results/transformer_rescue",
        "results/cmf_tables",
        "results/cmf_diagnostics",
    ]
    files = []
    for root in inventory_roots:
        rp = Path(root)
        if rp.exists():
            files.extend(str(x) for x in sorted(rp.glob("**/*")) if x.is_file())
    (OUT / "input_inventory.txt").write_text("\n".join(files) + "\n", encoding="utf-8")


def read_csv(name: str) -> pd.DataFrame:
    path = SOURCES[name]
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def fmt_tex(df: pd.DataFrame, path: Path) -> None:
    path.write_text(df.to_latex(index=False, float_format=lambda x: f"{x:.4f}" if pd.notna(x) else "NA"), encoding="utf-8")


def write_table(df: pd.DataFrame, name: str) -> pd.DataFrame:
    csv_path = TABLES / f"{name}.csv"
    tex_path = TABLES / f"{name}.tex"
    df.to_csv(csv_path, index=False)
    fmt_tex(df, tex_path)
    return df


def safe_num(s, default=np.nan):
    return pd.to_numeric(s, errors="coerce") if isinstance(s, pd.Series) else pd.to_numeric(pd.Series(s), errors="coerce").iloc[0]


def metric_trap_from_counts(dataset: str, num_pos: float, num_neg: float, source: str, notes: str = "") -> dict:
    num_pos = float(num_pos) if pd.notna(num_pos) else np.nan
    num_neg = float(num_neg) if pd.notna(num_neg) else np.nan
    if not pd.notna(num_pos) or not pd.notna(num_neg) or num_pos + num_neg <= 0:
        return {
            "dataset": dataset,
            "num_pos": np.nan,
            "num_neg": np.nan,
            "positive_rate": np.nan,
            "predict_all_normal_accuracy": np.nan,
            "predict_all_normal_weighted_f1": np.nan,
            "predict_all_normal_normal_f1": np.nan,
            "predict_all_normal_attack_f1": np.nan,
            "predict_all_attack_attack_f1": np.nan,
            "imbalance_severity_score": np.nan,
            "metric_trap_risk_level": "unknown",
            "source": source,
            "notes": notes or "missing class counts",
        }
    yt = np.r_[np.ones(int(round(num_pos))), np.zeros(int(round(num_neg)))]
    all_normal = compute_binary_metrics(yt, np.zeros_like(yt))
    all_attack = compute_binary_metrics(yt, np.ones_like(yt))
    pos_rate = num_pos / max(num_pos + num_neg, 1)
    severity = -math.log10(max(pos_rate, 1e-12))
    if pos_rate < 0.005:
        risk = "critical"
    elif pos_rate < 0.02:
        risk = "high"
    elif pos_rate < 0.05:
        risk = "medium"
    else:
        risk = "low"
    return {
        "dataset": dataset,
        "num_pos": int(round(num_pos)),
        "num_neg": int(round(num_neg)),
        "positive_rate": pos_rate,
        "predict_all_normal_accuracy": all_normal["accuracy"],
        "predict_all_normal_weighted_f1": all_normal["weighted_f1"],
        "predict_all_normal_normal_f1": all_normal["normal_f1"],
        "predict_all_normal_attack_f1": all_normal["attack_f1"],
        "predict_all_attack_attack_f1": all_attack["attack_f1"],
        "imbalance_severity_score": severity,
        "metric_trap_risk_level": risk,
        "source": source,
        "notes": notes,
    }


def make_metric_trap_audit() -> pd.DataFrame:
    rows = []
    ds = read_csv("dataset_summary")
    if not ds.empty:
        for _, r in ds.iterrows():
            windows = pd.to_numeric(pd.Series([r.get("Windows")]), errors="coerce").iloc[0]
            ratio = pd.to_numeric(pd.Series([r.get("Attack ratio")]), errors="coerce").iloc[0]
            if pd.notna(windows) and pd.notna(ratio):
                pos = windows * ratio
                neg = windows - pos
            else:
                pos = neg = np.nan
            rows.append(metric_trap_from_counts(str(r.get("Dataset")), pos, neg, "paper_table_dataset_summary_refined", str(r.get("Notes", ""))))
    full = read_csv("table13_metrics")
    if not full.empty:
        r = full.iloc[0]
        rows.append(metric_trap_from_counts("ctt_test04_sample_level", r.get("num_pos"), r.get("num_neg"), "metric_correction_full_metrics", "sample-level class totals"))
    out = pd.DataFrame(rows).drop_duplicates("dataset")
    write_table(out, "b1_metric_trap_audit_all_datasets")
    most = out.sort_values("positive_rate").head(5)
    (OUT / "b1_metric_trap_audit_all_datasets.md").write_text(
        "# Metric Trap Audit Across Datasets\n\n"
        "The trap is not unique to CT&T test04, but CT&T test04 is among the most extreme rare-attack settings. "
        "Rows with very low positive rate allow all-normal predictions to obtain high accuracy/weighted-F1 while attack-F1 remains zero.\n\n"
        f"Most severe rows:\n\n```csv\n{most.to_csv(index=False)}```\n",
        encoding="utf-8",
    )
    return out


def make_ctt_corrected_benchmark(metric_trap: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grain = read_csv("ctt_grain")
    if not grain.empty:
        for _, r in grain[grain["dataset"].astype(str).str.startswith("ctt_test")].iterrows():
            rows.append({
                "dataset": r["dataset"],
                "setting": r["dataset"],
                "model": f"{r.get('model')} / {r.get('granularity')}",
                "model_family": "GRAIN-CAN",
                "granularity": r.get("granularity"),
                "accuracy": r.get("accuracy", np.nan),
                "weighted_f1": np.nan,
                "attack_precision": r.get("precision", np.nan),
                "attack_recall": r.get("recall", np.nan),
                "attack_f1": r.get("f1", np.nan),
                "macro_f1": r.get("macro_f1", np.nan),
                "balanced_accuracy": np.nan,
                "mcc": np.nan,
                "auroc": r.get("auroc", np.nan),
                "aupr": r.get("aupr", np.nan),
                "recall_at_fpr_1e_4": r.get("recall_at_fpr_1em04", np.nan),
                "recall_at_fpr_5e_4": r.get("recall_at_fpr_5em04", np.nan),
                "recall_at_fpr_1e_3": r.get("recall_at_fpr_1em03", np.nan),
                "recall_at_fpr_5e_3": np.nan,
                "recall_at_fpr_1e_2": np.nan,
                "source": "final_grain_can_b1_granularity_full_matrix",
            })
    sample = read_csv("sample_ml")
    if not sample.empty:
        use = sample[sample["dataset"].astype(str).str.startswith("ctt_test")]
        # Keep one row per setting/model/protocol family: 5x when present, else capped.
        order = {"C_5x_negative_cap": 0, "B_2x_negative_cap": 1, "A_capped_current": 2, "A_capped": 3}
        use = use.assign(_ord=use["negative_protocol"].map(order).fillna(9))
        use = use.sort_values(["dataset", "model", "_ord", "seed"]).drop_duplicates(["dataset", "model"], keep="first")
        for _, r in use.iterrows():
            rows.append({
                "dataset": r["dataset"],
                "setting": r["dataset"],
                "model": f"{r.get('model')} / sample",
                "model_family": "Table13-style ML",
                "granularity": "sample",
                "accuracy": r.get("accuracy", np.nan),
                "weighted_f1": np.nan,
                "attack_precision": r.get("precision", np.nan),
                "attack_recall": r.get("recall", np.nan),
                "attack_f1": r.get("f1", np.nan),
                "macro_f1": r.get("macro_f1", np.nan),
                "balanced_accuracy": np.nan,
                "mcc": np.nan,
                "auroc": r.get("auroc", np.nan),
                "aupr": r.get("aupr", np.nan),
                "recall_at_fpr_1e_4": np.nan,
                "recall_at_fpr_5e_4": np.nan,
                "recall_at_fpr_1e_3": np.nan,
                "recall_at_fpr_5e_3": r.get("recall_at_fpr_5em03", np.nan),
                "recall_at_fpr_1e_2": r.get("recall_at_fpr_1em02", np.nan),
                "source": "final_grain_can_a1_negative_stability",
            })
    main = read_csv("main_results")
    if not main.empty:
        for _, r in main[main["Dataset/Setting"].astype(str).str.contains("CT&T", na=False)].iterrows():
            setting = "ctt_" + str(r["Dataset/Setting"]).split()[-1].lower()
            rows.append({
                "dataset": setting,
                "setting": setting,
                "model": r.get("Model"),
                "model_family": "window100 deep baseline",
                "granularity": "window100_deep",
                "accuracy": np.nan,
                "weighted_f1": np.nan,
                "attack_precision": r.get("precision", np.nan),
                "attack_recall": r.get("recall", np.nan),
                "attack_f1": r.get("f1", np.nan),
                "macro_f1": r.get("macro_f1", np.nan),
                "balanced_accuracy": np.nan,
                "mcc": np.nan,
                "auroc": r.get("auroc", np.nan),
                "aupr": r.get("aupr", np.nan),
                "recall_at_fpr_1e_4": np.nan,
                "recall_at_fpr_5e_4": np.nan,
                "recall_at_fpr_1e_3": np.nan,
                "recall_at_fpr_5e_3": np.nan,
                "recall_at_fpr_1e_2": np.nan,
                "source": "cmf_tables_overall_main_results_refined",
            })
    for _, r in metric_trap[metric_trap["dataset"].astype(str).isin(CTT_SETTINGS)].iterrows():
        for name, attack_f1, weighted in [
            ("predict_all_normal", 0.0, r.get("predict_all_normal_weighted_f1", np.nan)),
            ("predict_all_attack", r.get("predict_all_attack_attack_f1", np.nan), np.nan),
        ]:
            rows.append({
                "dataset": r["dataset"],
                "setting": r["dataset"],
                "model": name,
                "model_family": "trivial",
                "granularity": "none",
                "accuracy": r.get("predict_all_normal_accuracy") if name == "predict_all_normal" else r.get("positive_rate"),
                "weighted_f1": weighted,
                "attack_precision": np.nan,
                "attack_recall": 0.0 if name == "predict_all_normal" else 1.0,
                "attack_f1": attack_f1,
                "macro_f1": np.nan,
                "balanced_accuracy": 0.5,
                "mcc": 0.0,
                "auroc": np.nan,
                "aupr": r.get("positive_rate"),
                "recall_at_fpr_1e_4": 0.0 if name == "predict_all_normal" else np.nan,
                "recall_at_fpr_5e_4": 0.0 if name == "predict_all_normal" else np.nan,
                "recall_at_fpr_1e_3": 0.0 if name == "predict_all_normal" else np.nan,
                "recall_at_fpr_5e_3": 0.0 if name == "predict_all_normal" else np.nan,
                "recall_at_fpr_1e_2": 0.0 if name == "predict_all_normal" else np.nan,
                "source": "derived_trivial_baseline",
            })
    out = pd.DataFrame(rows)
    counts = metric_trap[["dataset", "num_pos", "num_neg", "positive_rate"]].rename(columns={"dataset": "setting"})
    out = out.merge(counts, on="setting", how="left")
    out = out.sort_values(["setting", "attack_f1"], ascending=[True, False])
    write_table(out, "c1_ctt_all_settings_corrected_benchmark")
    best = out.groupby("setting", dropna=False).head(1)
    (OUT / "c1_ctt_all_settings_corrected_benchmark.md").write_text(
        "# CT&T All-Settings Corrected Benchmark\n\n"
        "Corrected metrics show CT&T test04 remains the hardest shifted setting. GRAIN aggregate/window features dominate the corrected test04 attack-F1 leaderboard, while weighted-F1 can still make trivial baselines look strong.\n\n"
        f"Best row per setting:\n\n```csv\n{best.to_csv(index=False)}```\n",
        encoding="utf-8",
    )
    return out


def make_table13_case() -> pd.DataFrame:
    targets = read_csv("table13_targets")
    match = read_csv("table13_match")
    rows = []
    def first_mode(series: pd.Series, default: str = "unknown") -> str:
        if series is None:
            return default
        m = series.dropna().mode()
        return str(m.iloc[0]) if len(m) else default

    for _, t in targets.iterrows():
        sub = match[match["model"].astype(str).eq(str(t.get("model")))]
        hyp = first_mode(sub["overall_metric_hypothesis"], "unknown") if not sub.empty and "overall_metric_hypothesis" in sub else "unknown"
        best_f1 = first_mode(sub["best_matching_f1_metric"], "unknown") if not sub.empty and "best_matching_f1_metric" in sub else "unknown"
        rows.append({
            "model": t.get("model"),
            "reported_accuracy": t.get("reported_accuracy"),
            "reported_recall": t.get("reported_recall"),
            "reported_f1": t.get("reported_f1"),
            "accuracy_equals_recall": bool(pd.notna(t.get("reported_accuracy")) and abs(t.get("reported_accuracy") - t.get("reported_recall")) < 1e-4),
            "best_matching_f1_metric": best_f1,
            "model_level_hypothesis": hyp,
            "source": "metric_correction_paper",
        })
    out = pd.DataFrame(rows)
    write_table(out, "d1_table13_case_study")
    eq = int(out["accuracy_equals_recall"].sum())
    (OUT / "d1_table13_case_study.md").write_text(
        "# Table 13 Case Study\n\n"
        f"{eq} rows have reported accuracy equal to reported recall within 1e-4. This supports a weighted/accuracy-like metric hypothesis, but it does not prove the original authors are wrong because their confusion matrices and exact code are unavailable.\n\n"
        "The safe statement is metric ambiguity plus corrected attack-centric benchmark requirement.\n",
        encoding="utf-8",
    )
    return out


def make_ranking(ctt: pd.DataFrame) -> pd.DataFrame:
    ranked, summary = compute_ranking_inversion(ctt, setting_cols=("setting",))
    # compute helper columns requested by the task
    helper = []
    for setting, g in ranked.groupby("setting"):
        row = {"setting": setting}
        pan = g[g["model"].eq("predict_all_normal")]
        if len(pan):
            row["predict_all_normal_rank_by_weighted"] = pan["rank_by_weighted_f1"].iloc[0]
            row["predict_all_normal_rank_by_attack"] = pan["rank_by_attack_f1"].iloc[0]
        grain = g[g["model"].astype(str).str.contains("GRAIN_window_100|window_100", na=False)]
        if len(grain):
            row["grain_rank_by_attack"] = grain["rank_by_attack_f1"].min()
            row["grain_rank_by_weighted"] = grain["rank_by_weighted_f1"].min()
        helper.append(row)
    out = summary.merge(pd.DataFrame(helper), on="setting", how="left")
    write_table(out, "e1_ranking_inversion_all")
    (OUT / "e1_ranking_inversion_all.md").write_text(
        "# Ranking Inversion Across Settings\n\n"
        "Ranking by weighted-F1 and attack-F1 changes model selection in rare-attack settings. The all-normal baseline can rank high under weighted-F1 while ranking last by attack-F1.\n",
        encoding="utf-8",
    )
    return out


def make_low_fpr(ctt: pd.DataFrame) -> pd.DataFrame:
    low = read_csv("low_fpr")
    rows = []
    if not low.empty:
        for _, r in low.iterrows():
            rows.append({
                "dataset": r.get("dataset"),
                "model": r.get("model"),
                "granularity": r.get("granularity"),
                "threshold_type": r.get("threshold_type"),
                "fpr_budget": r.get("fpr_budget"),
                "recall_at_fpr": r.get("recall_at_fpr"),
                "precision_at_fpr": r.get("precision_at_fpr"),
                "f1_at_fpr": r.get("f1_at_fpr"),
                "actual_fpr": r.get("actual_fpr"),
                "aupr": np.nan,
                "auroc": np.nan,
                "source": "final_grain_can_e1_low_fpr",
            })
    for _, r in ctt[ctt["model"].eq("predict_all_normal")].iterrows():
        for b in [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]:
            rows.append({
                "dataset": r["dataset"],
                "model": "predict_all_normal",
                "granularity": "none",
                "threshold_type": "formal_trivial",
                "fpr_budget": b,
                "recall_at_fpr": 0.0,
                "precision_at_fpr": 0.0,
                "f1_at_fpr": 0.0,
                "actual_fpr": 0.0,
                "aupr": r.get("aupr", np.nan),
                "auroc": np.nan,
                "source": "derived_trivial_baseline",
            })
    out = pd.DataFrame(rows)
    write_table(out, "f1_low_fpr_deployment")
    (OUT / "f1_low_fpr_deployment.md").write_text(
        "# Low-FPR Deployment Evaluation\n\n"
        "High weighted-F1 does not imply useful low-FPR recall. GRAIN aggregate rows provide the strongest available low-FPR evidence, but test04 remains difficult and cannot be described as solved.\n",
        encoding="utf-8",
    )
    return out


def make_event(ctt: pd.DataFrame) -> pd.DataFrame:
    event = read_csv("event")
    rows = []
    if not event.empty:
        for _, r in event.iterrows():
            rows.append({
                "dataset": r.get("dataset"),
                "model": r.get("model"),
                "granularity": r.get("granularity"),
                "event_recall": r.get("event_recall"),
                "mean_detection_delay": r.get("mean_detection_delay_seconds"),
                "median_detection_delay": r.get("median_detection_delay_seconds"),
                "false_alarm_per_100k": np.nan,
                "false_alarm_per_hour": r.get("false_alarm_samples_per_hour"),
                "recall_under_FA_budget": r.get("recall_at_1_FA_per_hour", np.nan),
                "event_boundary_source": r.get("event_boundary_quality", "approximate_from_labels"),
                "source": "final_grain_can_f2_event_level",
            })
    for _, r in ctt[ctt["model"].eq("predict_all_normal")].iterrows():
        rows.append({
            "dataset": r["dataset"],
            "model": "predict_all_normal",
            "granularity": "none",
            "event_recall": 0.0,
            "mean_detection_delay": np.nan,
            "median_detection_delay": np.nan,
            "false_alarm_per_100k": 0.0,
            "false_alarm_per_hour": 0.0,
            "recall_under_FA_budget": 0.0,
            "event_boundary_source": "derived_trivial",
            "source": "derived_trivial_baseline",
        })
    out = pd.DataFrame(rows)
    write_table(out, "g1_event_level_evaluation")
    (OUT / "g1_event_level_evaluation.md").write_text(
        "# Event-Level Evaluation\n\n"
        "Event-level evaluation is closer to IDS deployment than isolated sample/window metrics. Available CT&T event boundaries are approximate from labels; paper language must be conservative.\n",
        encoding="utf-8",
    )
    return out


def make_feature_analysis() -> pd.DataFrame:
    feat = read_csv("feature")
    grain = read_csv("ctt_grain")
    rows = []
    if not feat.empty:
        rows.extend(feat.to_dict("records"))
    if not grain.empty:
        for _, r in grain[grain["dataset"].eq("ctt_test04")].iterrows():
            rows.append({
                "granularity": r.get("granularity"),
                "feature": "all_feature_preserving_aggregate",
                "single_feature_auc": np.nan,
                "tree_feature_importance": np.nan,
                "effect_size": np.nan,
                "mutual_information": np.nan,
                "permutation_importance": np.nan,
                "train_test_shift": np.nan,
                "feature_missing_rate": np.nan,
                "aggregation_distortion": "summary_row",
                "leakage_risk": "audited",
                "attack_f1": r.get("f1"),
                "aupr": r.get("aupr"),
                "recall_at_fpr_1e_3": r.get("recall_at_fpr_1em03", np.nan),
            })
    out = pd.DataFrame(rows)
    write_table(out, "h1_grain_feature_granularity_analysis")
    (OUT / "h1_grain_feature_granularity_analysis.md").write_text(
        "# GRAIN Feature and Granularity Analysis\n\n"
        "Strong signals come from causal timing and payload-preserving features such as delta_t_same_id, payload_delta_l1, payload_sum/std, bytes and CAN ID. Aggregate window_100 preserves these summaries, unlike old deep window pipelines that can dilute rare evidence.\n",
        encoding="utf-8",
    )
    return out


def make_external_sanity(metric_trap: pd.DataFrame) -> pd.DataFrame:
    ext = read_csv("external")
    rows = []
    if not ext.empty:
        for _, r in ext.iterrows():
            ds = r.get("dataset")
            if pd.isna(ds):
                continue
            rows.append({
                "dataset": ds,
                "model": r.get("model", r.get("Model", "NA")),
                "positive_rate": np.nan,
                "predict_all_normal_weighted_f1": np.nan,
                "predict_all_normal_attack_f1": 0.0,
                "best_available_model_attack_f1": r.get("f1", np.nan),
                "best_available_model_aupr": r.get("aupr", np.nan),
                "recall_at_fpr_1e_3": r.get("recall_at_fpr_1em03", np.nan),
                "notes_on_protocol": r.get("scope", r.get("source", "")),
                "source": "final_grain_can_g1_cross_dataset_validation",
            })
    for _, r in metric_trap[~metric_trap["dataset"].astype(str).str.startswith("ctt_test")].iterrows():
        rows.append({
            "dataset": r["dataset"],
            "model": "predict_all_normal",
            "positive_rate": r.get("positive_rate"),
            "predict_all_normal_weighted_f1": r.get("predict_all_normal_weighted_f1"),
            "predict_all_normal_attack_f1": 0.0,
            "best_available_model_attack_f1": np.nan,
            "best_available_model_aupr": np.nan,
            "recall_at_fpr_1e_3": np.nan,
            "notes_on_protocol": r.get("notes", ""),
            "source": "metric_trap_audit",
        })
    out = pd.DataFrame(rows).drop_duplicates(["dataset", "model"], keep="first")
    write_table(out, "i1_external_corrected_sanity")
    (OUT / "i1_external_corrected_sanity.md").write_text(
        "# External Corrected Sanity Check\n\n"
        "External datasets support attack-centric reporting, but some are subsets or sanity checks. CT&T test04 remains the most extreme metric-trap case in the current evidence package.\n",
        encoding="utf-8",
    )
    return out


def plot_bar(df, path: Path, x: str, y: str, title: str, hue: str | None = None, top_n: int | None = None) -> None:
    d = df.copy()
    d[y] = pd.to_numeric(d[y], errors="coerce")
    d = d.dropna(subset=[x, y])
    if top_n:
        d = d.sort_values(y, ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    if d.empty:
        ax.text(0.5, 0.5, "No supported data", ha="center", va="center")
    elif hue and hue in d:
        piv = d.pivot_table(index=x, columns=hue, values=y, aggfunc="max")
        piv.plot(kind="bar", ax=ax, edgecolor="#333333", linewidth=0.5, color=PLOT_COLORS[: len(piv.columns)])
    else:
        colors = [PLOT_COLORS[i % len(PLOT_COLORS)] for i in range(len(d))]
        ax.bar(range(len(d)), d[y], color=colors, edgecolor="#333333", linewidth=0.5)
        ax.set_xticks(range(len(d)))
        ax.set_xticklabels(d[x].astype(str), rotation=35, ha="right")
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(y)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.6)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, format="svg")
    plt.close(fig)


def plot_line(df, path: Path, x: str, y: str, title: str, group: str) -> None:
    d = df.copy()
    d[x] = pd.to_numeric(d[x], errors="coerce")
    d[y] = pd.to_numeric(d[y], errors="coerce")
    d = d.dropna(subset=[x, y])
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    if d.empty:
        ax.text(0.5, 0.5, "No supported data", ha="center", va="center")
    else:
        markers = ["o", "s", "^", "D", "v", "P"]
        styles = ["-", "--", "-.", ":"]
        for i, (name, g) in enumerate(d.groupby(group)):
            g = g.sort_values(x)
            ax.plot(g[x], g[y], marker=markers[i % len(markers)], linestyle=styles[i % len(styles)], linewidth=1.4, markersize=4, label=str(name))
        ax.legend(frameon=False, fontsize=8, loc="best")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.6)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, format="svg")
    plt.close(fig)


def plot_pipeline(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 2.2))
    ax.axis("off")
    boxes = [
        ("Raw CAN\nframes", 0.08),
        ("Attack-centric\nmetrics", 0.28),
        ("Metric\nforensics", 0.48),
        ("Corrected\nbenchmark", 0.68),
        ("GRAIN-CAN\nbaseline", 0.88),
    ]
    for txt, x in boxes:
        ax.text(x, 0.55, txt, ha="center", va="center", fontsize=9, bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="black", lw=0.8))
    for i in range(len(boxes) - 1):
        ax.annotate("", xy=(boxes[i + 1][1] - 0.08, 0.55), xytext=(boxes[i][1] + 0.08, 0.55), arrowprops=dict(arrowstyle="->", lw=0.9, color="#333333"))
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)


def make_figures(metric_trap, ctt, case, ranking, low, event, feat, external) -> None:
    plot_bar(metric_trap, FIGS / "paper_fig1_metric_trap_across_datasets.svg", "dataset", "predict_all_normal_weighted_f1", "Metric Trap Across Datasets", top_n=12)
    plot_bar(ctt[ctt["model_family"].ne("trivial")], FIGS / "paper_fig2_ctt_corrected_benchmark.svg", "model", "attack_f1", "CT&T Corrected Benchmark", hue="setting", top_n=24)
    plot_bar(case, FIGS / "paper_fig3_table13_case_study.svg", "model", "accuracy_equals_recall", "Table 13 Case Study", top_n=16)
    plot_bar(ranking, FIGS / "paper_fig4_ranking_inversion.svg", "setting", "top3_overlap_weighted_vs_attack", "Ranking Inversion", top_n=10)
    plot_line(low, FIGS / "paper_fig5_low_fpr_deployment.svg", "fpr_budget", "recall_at_fpr", "Low-FPR Deployment", "dataset")
    plot_bar(event, FIGS / "paper_fig6_event_level.svg", "dataset", "event_recall", "Event-Level Recall", hue="model")
    plot_bar(feat, FIGS / "paper_fig7_grain_feature_preservation.svg", "feature", "single_feature_auc", "Feature Preservation", top_n=14)
    plot_bar(external, FIGS / "paper_fig8_external_sanity.svg", "dataset", "best_available_model_attack_f1", "External Corrected Sanity", top_n=14)
    plot_pipeline(FIGS / "paper_fig9_attack_centric_pipeline.svg")


def make_paper_tables(metric_trap, ctt, low, event, feat, external) -> None:
    ds = read_csv("dataset_summary")
    write_table(ds, "paper_table1_dataset_stats")
    write_table(metric_trap, "paper_table2_metric_trap_audit")
    write_table(ctt, "paper_table3_corrected_ctt_benchmark")
    write_table(low, "paper_table4_low_fpr")
    write_table(event, "paper_table5_event_level")
    write_table(feat, "paper_table6_grain_ablation")
    write_table(external, "paper_table7_external_sanity")


def write_toolkit_doc() -> None:
    (TOOLKIT / "attack_centric_eval_api.md").write_text(
        "# Attack-Centric Evaluation API\n\n"
        "`cmf_can.analysis.attack_centric_eval` provides:\n"
        "- `compute_binary_metrics(y_true, y_pred, attack_label=1)` for attack/normal precision, recall, F1, macro-F1, weighted-F1, accuracy, balanced accuracy, MCC and confusion matrix.\n"
        "- `compute_score_metrics(y_true, y_score)` for AUROC, AUPR and Recall/Precision/F1@FPR budgets 1e-4, 5e-4, 1e-3, 5e-3, 1e-2.\n"
        "- `compute_trivial_baselines(y_true)` for all-normal, all-attack and random rare-attack baselines.\n"
        "- `compute_ranking_inversion(metrics_df)` for rank disagreement, Spearman/Kendall and top-k overlap.\n"
        "- `compute_event_metrics(prediction_df)` for approximate event recall, delay and false alarms.\n\n"
        "In single-label classification, weighted recall equals accuracy because each class recall is weighted by class support, so the numerator becomes total correct predictions. In rare-attack IDS, this can hide a zero attack-F1 detector. Therefore papers must report attack-positive precision/recall/F1, AUPR, low-FPR recall and event-level evidence before claiming an IDS is effective.\n",
        encoding="utf-8",
    )


def write_final_docs(metric_trap, ctt, ranking) -> None:
    best_test04 = ctt[ctt["setting"].eq("ctt_test04")].sort_values("attack_f1", ascending=False).head(1)
    best_line = best_test04.iloc[0].to_dict() if len(best_test04) else {}
    docs = {
        "final_paper_theme.md": "# Final Paper Theme\n\n中文题目：面向跨车未知攻击的车载 CAN 入侵检测：攻击中心化评估与特征保真检测方法\n\nEnglish title: Attack-Centric Evaluation and Feature-Preserving Detection for Cross-Vehicle Unknown-Attack CAN IDS\n\nPaper type: Research Paper\n\nMain line: attack-centric evaluation framework + corrected benchmark + feature-preserving IDS baseline. It is not only a CT&T critique and not ordinary model tuning.\n",
        "final_contributions.md": "# Final Contributions\n\n1. Attack-centric evaluation framework for rare-attack CAN IDS.\n2. Metric forensics and imbalance-trap analysis showing why weighted/accuracy-like metrics can mislead.\n3. Corrected multi-setting and multi-dataset benchmark with attack-F1, AUPR, low-FPR and event-level evidence.\n4. GRAIN-CAN feature-preserving strong baseline for corrected CT&T shifted evaluation.\n",
        "recommended_paper_outline.md": "# Recommended Paper Outline\n\nTitle: Attack-Centric Evaluation and Feature-Preserving Detection for Cross-Vehicle Unknown-Attack CAN IDS\n\n1. Introduction: rare attack evaluation can be misleading.\n2. Background: CT&T shifted CAN IDS and metric ambiguity.\n3. Attack-centric evaluation framework.\n4. Table 13 metric forensics case study.\n5. Corrected CT&T test01-test04 benchmark.\n6. GRAIN-CAN feature-preserving baseline.\n7. Low-FPR and event-level deployment evidence.\n8. External sanity checks.\n9. Threats to validity and reporting recommendations.\n",
        "main_claims.md": f"# Main Claims\n\n- Weighted/accuracy-like metrics can make all-normal or weak detectors look strong in rare-attack CAN IDS.\n- CT&T test04 is an extreme example, but the metric trap appears across rare-attack settings.\n- Corrected attack-centric metrics show test04 is not solved.\n- Best current corrected test04 row: `{best_line.get('model','NA')}` with attack-F1={best_line.get('attack_f1', np.nan):.4f}, AUPR={best_line.get('aupr', np.nan):.4f}.\n- GRAIN-CAN is a strong corrected baseline, not a final unknown-attack solution.\n",
        "unsafe_claims_do_not_write.md": "# Unsafe Claims Do Not Write\n\n- Do not write that the original paper is wrong.\n- Do not write that the CT&T dataset is bad.\n- Do not write that public 0.998 is attack-F1.\n- Do not directly compare attack-F1 and weighted-F1.\n- Do not write that unknown attack is solved.\n- Do not write that GRAIN-CAN is ultimate SOTA.\n- Do not write that predict_all_normal is an IDS.\n",
        "threats_to_validity.md": "# Threats To Validity\n\n- Original paper confusion matrices and exact code are unavailable.\n- Some event-level metrics use approximate boundaries from labels.\n- External datasets include subsets and sanity-check datasets.\n- Some low-FPR rows are best-test upper bounds where validation score dumps are unavailable.\n- The corrected benchmark should be presented as evidence-backed but not as a definitive final standard without community validation.\n",
        "security4_readiness.md": "# Security4 Readiness\n\nCurrent package is credible as a measurement + IDS baseline research paper direction. It is stronger than ordinary model tuning because it provides metric forensics, corrected benchmark, low-FPR and event-level evidence.\n\nIt is not yet a clean unknown-attack breakthrough paper. For CCF A / Security Four, the main remaining gaps are original-author confusion matrices or code, exact v1.5 alignment, official event boundaries, and broader external full-dataset validation. CIVS-style research paper readiness is stronger. The safest framing is measurement plus corrected IDS baseline.\n",
    }
    for name, text in docs.items():
        (OUT / name).write_text(text, encoding="utf-8")
    inventory = [
        "| Item | Path | Source | Recommended placement | Caveat |",
        "|---|---|---|---|---|",
    ]
    for fig in sorted(FIGS.glob("paper_fig*.svg")):
        inventory.append(f"| Figure | `{fig}` | corresponding CSV in tables/ | main/appendix by paper length | SVG only |")
    for tbl in sorted(TABLES.glob("paper_table*.csv")):
        inventory.append(f"| Table | `{tbl}` | generated from existing result CSVs | main/appendix | NA retained where unsupported |")
    (OUT / "paper_figure_table_inventory.md").write_text("\n".join(inventory) + "\n", encoding="utf-8")


def main() -> None:
    setup()
    write_toolkit_doc()
    metric_trap = make_metric_trap_audit()
    ctt = make_ctt_corrected_benchmark(metric_trap)
    case = make_table13_case()
    ranking = make_ranking(ctt)
    low = make_low_fpr(ctt)
    event = make_event(ctt)
    feat = make_feature_analysis()
    external = make_external_sanity(metric_trap)
    make_figures(metric_trap, ctt, case, ranking, low, event, feat, external)
    make_paper_tables(metric_trap, ctt, low, event, feat, external)
    write_final_docs(metric_trap, ctt, ranking)


if __name__ == "__main__":
    main()
