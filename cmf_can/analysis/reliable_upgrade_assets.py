from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(".")
SRC_TABLES = ROOT / "results/cmf_tables"
SRC_PREDS = ROOT / "results/cmf_predictions"
SRC_EMB = ROOT / "results/cmf_embeddings"
OUT = ROOT / "results/top_tier_upgrade"
PRED_OUT = OUT / "predictions"
FIG_OUT = OUT / "figures"


def setup_style() -> None:
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


def savefig(fig: plt.Figure, name: str) -> None:
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIG_OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def csv(name: str) -> pd.DataFrame:
    path = SRC_TABLES / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def write_tex(df: pd.DataFrame, path: Path) -> None:
    path.write_text(df.to_latex(index=False, escape=True, na_rep="NA", float_format=lambda x: f"{x:.4f}"), encoding="utf-8")


def clean_model(model: str) -> str:
    return {
        "cmf_can": "CMF-CAN",
        "reliable_cmf_can": "Reliable-CMF-CAN",
        "concat_fusion": "Concat-Fusion",
        "transformer": "Transformer",
        "frame_only": "Frame-only",
        "stats_only": "Stats-only",
        "wo_context": "-Ctx",
        "wo_stats": "-Win",
        "wo_gate": "-Gate",
        "wo_xattn": "-XAttn",
        "cmf_can_robust": "Robust CMF",
    }.get(model, model)


def copy_prediction_assets() -> tuple[pd.DataFrame, list[str]]:
    PRED_OUT.mkdir(parents=True, exist_ok=True)
    required = []
    for ds in ["road", "ctt_test01", "ctt_test02", "ctt_test03", "ctt_test04"]:
        models = ["transformer", "concat_fusion", "cmf_can"]
        if ds.startswith("ctt"):
            models += ["frame_only", "stats_only", "wo_context", "wo_stats"]
        for model in models:
            required.append((ds, model, "prediction"))
        if ds in ["road", "ctt_test01", "ctt_test02", "ctt_test03", "ctt_test04"]:
            required.append((ds, "cmf_can", "gate"))
    rows = []
    missing = []
    for ds, model, kind in required:
        if kind == "prediction":
            src = SRC_PREDS / f"{ds}_{model}_predictions.csv"
            dst = PRED_OUT / f"{ds}_{ds}_{model}_predictions.csv"
        else:
            src = SRC_PREDS / f"{ds}_{model}_gate_weights.csv"
            dst = PRED_OUT / f"{ds}_{ds}_CMF-CAN_gate_weights.csv"
        exists = src.exists()
        if exists:
            shutil.copy2(src, dst)
        else:
            missing.append(f"missing {kind}: {ds}/{model}")
        rows.append({"dataset": ds, "setting": ds, "model": model, "artifact_type": kind, "source": str(src), "output": str(dst) if exists else "NA", "status": "available" if exists else "missing"})
    for emb in sorted(SRC_EMB.glob("*_embedding_sample.npy")):
        dst = PRED_OUT / emb.name.replace("_embedding_sample.npy", "_embeddings_sample.npy")
        shutil.copy2(emb, dst)
        rows.append({"dataset": emb.name.split("_cmf_can_", 1)[0], "setting": emb.name.split("_cmf_can_", 1)[0], "model": "cmf_can", "artifact_type": "embedding_sample", "source": str(emb), "output": str(dst), "status": "sample_only"})
    inv = pd.DataFrame(rows)
    inv.to_csv(OUT / "01_prediction_gate_embedding_inventory.csv", index=False)
    lines = ["# Prediction / Gate / Embedding Inventory", "", "| Dataset | Model | Type | Status |", "|---|---|---|---|"]
    for r in inv.to_dict("records"):
        lines.append(f"| {r['dataset']} | {r['model']} | {r['artifact_type']} | {r['status']} |")
    if missing:
        lines += ["", "## Missing", ""]
        lines += [f"- {m}" for m in missing]
    (OUT / "01_prediction_gate_embedding_inventory.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return inv, missing


def current_limitations() -> list[str]:
    main = csv("paper_table_overall_main_results_refined.csv")
    ab = csv("paper_table_ablation_refined.csv")
    gen = csv("ctt_generalization_15ep.csv")
    low = csv("paper_table_low_fpr_refined.csv")
    few = csv("paper_table_few_label_refined.csv")
    rows = []
    notes = []
    if not main.empty:
        f1_col = "F1" if "F1" in main.columns else "f1"
        for dataset, g in main.groupby("Dataset/Setting"):
            cmf = g[g["Model"].astype(str).str.contains("CMF", case=False, na=False)]
            tr = g[g["Model"].astype(str).str.contains("Transformer", case=False, na=False)]
            if not cmf.empty and not tr.empty:
                cf = pd.to_numeric(cmf[f1_col], errors="coerce").max()
                tf = pd.to_numeric(tr[f1_col], errors="coerce").max()
                if pd.notna(cf) and pd.notna(tf) and cf < tf:
                    rows.append({"issue": "CMF below Transformer", "setting": dataset, "evidence": f"CMF F1={cf:.4f}, Transformer F1={tf:.4f}", "severity": "high"})
    if not ab.empty:
        for dataset, g in ab.groupby("Dataset"):
            full = g[g["Variant"].eq("Full")]
            if full.empty:
                continue
            full_f1 = float(full["F1"].iloc[0])
            better = g[pd.to_numeric(g["F1"], errors="coerce") > full_f1 + 1e-12]
            for _, r in better.iterrows():
                rows.append({"issue": "Ablation variant exceeds Full", "setting": dataset, "evidence": f"{r['Variant']} F1={float(r['F1']):.4f} > Full {full_f1:.4f}", "severity": "high"})
    if not gen.empty:
        for ds in ["ctt_test02", "ctt_test03", "ctt_test04"]:
            part = gen[gen["dataset"].eq(ds)]
            cmf = part[part["model"].eq("cmf_can")]
            if not cmf.empty:
                rows.append({"issue": "Shifted CT&T remains weak or threshold-sensitive", "setting": ds, "evidence": f"CMF F1={float(cmf['f1'].iloc[0]):.4f}, AUPR={float(cmf['aupr'].iloc[0]):.4f}", "severity": "high"})
    if not low.empty:
        best = low.sort_values("Recall", ascending=False).head(3) if "Recall" in low.columns else pd.DataFrame()
        for _, r in best.iterrows():
            rows.append({"issue": "Low-FPR bright spot", "setting": str(r.get("Dataset/Setting", r.get("dataset", "NA"))), "evidence": f"Recall={r.get('Recall', 'NA')} at {r.get('FPR Budget', r.get('fpr_budget', 'NA'))}", "severity": "opportunity"})
    if not few.empty:
        rows.append({"issue": "Few-label requires variance-aware claims", "setting": "ROAD/CT&T", "evidence": "few-label tables exist, but full stability depends on multi-seed setting-specific results", "severity": "medium"})
    rows += [
        {"issue": "Top-tier blocker", "setting": "CT&T test04", "evidence": "unknown vehicle + unknown attack remains weak under strict false-alarm constraints", "severity": "critical"},
        {"issue": "Top-tier opportunity", "setting": "CT&T test03/test02", "evidence": "normality and context-control evidence suggest reliable/adaptive policy can repair specific shifts", "severity": "opportunity"},
    ]
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "00_current_limitations.csv", index=False)
    lines = ["# Current Limitations", ""]
    for r in rows:
        lines.append(f"- **{r['issue']}** ({r['setting']}): {r['evidence']} [{r['severity']}]")
    (OUT / "00_current_limitations.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return notes


def plot_grouped(df: pd.DataFrame, x_col: str, y_col: str, hue_col: str, name: str, ylabel: str) -> None:
    if df.empty:
        return
    x_vals = list(dict.fromkeys(df[x_col].astype(str)))
    hue_vals = list(dict.fromkeys(df[hue_col].astype(str)))
    x = np.arange(len(x_vals))
    width = min(0.8 / max(len(hue_vals), 1), 0.28)
    fig, ax = plt.subplots(figsize=(max(5.5, len(x_vals) * 1.25), 3.2))
    for i, hue in enumerate(hue_vals):
        vals = []
        for xv in x_vals:
            hit = df[(df[x_col].astype(str).eq(xv)) & (df[hue_col].astype(str).eq(hue))]
            vals.append(float(hit[y_col].iloc[0]) if not hit.empty and pd.notna(hit[y_col].iloc[0]) else np.nan)
        ax.bar(x + (i - (len(hue_vals) - 1) / 2) * width, vals, width, label=hue, edgecolor="#222", hatch="//" if "Reliable" in hue else None)
    ax.set_xticks(x)
    ax.set_xticklabels(x_vals, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, ncol=min(3, len(hue_vals)))
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)
    savefig(fig, name)


def e1_main(missing: list[str]) -> None:
    road = csv("road_main_20ep.csv")
    ctt = csv("ctt_unknown_ablation.csv")
    rows = []
    if not road.empty:
        for model in ["transformer", "frame_only", "stats_only", "concat_fusion", "cmf_can", "wo_context", "wo_stats"]:
            hit = road[road["model"].eq(model)]
            if not hit.empty:
                r = hit.iloc[0].to_dict()
                rows.append({**r, "setting": "road", "display_model": clean_model(model), "source": "road_main_20ep.csv"})
    if not ctt.empty:
        for ds in ["ctt_test01", "ctt_test02", "ctt_test04"]:
            part = ctt[ctt["dataset"].eq(ds)]
            for model in ["transformer", "frame_only", "stats_only", "concat_fusion", "cmf_can", "wo_context", "wo_stats", "wo_gate", "wo_xattn"]:
                hit = part[part["model"].eq(model)]
                if not hit.empty:
                    r = hit.iloc[0].to_dict()
                    rows.append({**r, "setting": ds, "display_model": clean_model(model), "source": "ctt_unknown_ablation.csv"})
    out = pd.DataFrame(rows)
    # Reliable-CMF-CAN is implemented but not trained; record missing rows.
    for setting in ["road", "ctt_test01", "ctt_test02", "ctt_test04"]:
        for model in ["Reliable-CMF-CAN", "Reliable-CMF-CAN w/o shift", "Reliable-CMF-CAN w/o normality", "Reliable-CMF-CAN w/o segment"]:
            out = pd.concat([out, pd.DataFrame([{"setting": setting, "display_model": model, "model": model, "f1": np.nan, "macro_f1": np.nan, "auroc": np.nan, "aupr": np.nan, "source": "not trained"}])], ignore_index=True)
            missing.append(f"E1 missing trained checkpoint/result: {setting}/{model}")
    out.to_csv(OUT / "e1_reliable_gate_main.csv", index=False)
    write_tex(out, OUT / "e1_reliable_gate_main.tex")
    plot_grouped(out[out["display_model"].isin(["Transformer", "Frame-only", "Stats-only", "Concat-Fusion", "CMF-CAN", "-Ctx", "-Win"])], "setting", "f1", "display_model", "fig_e1_reliable_gate_main", "F1")
    (OUT / "e1_reliable_gate_analysis.md").write_text(
        "# E1 Reliable Gate Main\n\nReliable-CMF-CAN code is implemented and registered, but trained checkpoints do not yet exist. Existing ablations show over-fusion: simplified variants can beat Full CMF-CAN, especially -Ctx under unknown vehicle settings. No Reliable-CMF-CAN result is fabricated.\n",
        encoding="utf-8",
    )


def e2_context(missing: list[str]) -> None:
    ab = csv("ctt_unknown_ablation.csv")
    shift = csv("results/cmf_diagnostics/tables/d3_id_context_shift.csv") if False else pd.DataFrame()
    rows = []
    if not ab.empty:
        for ds in ["ctt_test01", "ctt_test02", "ctt_test04"]:
            for model in ["cmf_can", "wo_context", "frame_only", "stats_only", "wo_gate"]:
                hit = ab[(ab["dataset"].eq(ds)) & (ab["model"].eq(model))]
                if not hit.empty:
                    r = hit.iloc[0].to_dict()
                    rows.append({**r, "setting": ds, "display_model": clean_model(model), "context_policy": "static/no control" if model != "wo_context" else "context removed"})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "e2_shift_context.csv", index=False)
    write_tex(out, OUT / "e2_shift_context.tex")
    plot_grouped(out, "setting", "f1", "display_model", "fig_e2_shift_context", "F1")
    (OUT / "e2_shift_context_analysis.md").write_text(
        "# E2 Shift-Aware Context\n\nExisting evidence supports the hypothesis that ID-context can hurt under unknown vehicle shift: `wo_context` is far stronger than Full on CT&T test02/test04 in single-seed ablation. Reliable shift control is implemented but not trained; direct comparison against -Ctx remains missing.\n",
        encoding="utf-8",
    )
    missing.append("E2 missing trained Reliable-CMF-CAN shift-control checkpoint and context_shift_score dump")


def e3_normality(missing: list[str]) -> None:
    an = csv("anomaly_ensemble_final_shifted_summary.csv")
    rows = []
    if not an.empty:
        for r in an.to_dict("records"):
            rows.append({"dataset": r["dataset"], "model": "CMF-CAN baseline", "f1": r["baseline_f1_mean"], "aupr": r["baseline_aupr_mean"], "recall_at_fpr_1em04": r["baseline_recall_at_fpr_1em04_mean"], "source": "anomaly_ensemble_final_shifted_summary.csv"})
            rows.append({"dataset": r["dataset"], "model": "CMF-CAN + normality policy", "f1": r["best_f1_mean"], "aupr": r["best_f1_aupr_mean"], "recall_at_fpr_1em04": r["best_lowfpr_recall_at_fpr_1em04_mean"], "source": "anomaly_ensemble_final_shifted_summary.csv"})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "e3_normality_unknown_attack.csv", index=False)
    write_tex(out, OUT / "e3_normality_unknown_attack.tex")
    plot_grouped(out, "dataset", "f1", "model", "fig_e3_normality_unknown_attack", "F1")
    (OUT / "e3_normality_unknown_attack_analysis.md").write_text(
        "# E3 Normality Unknown Attack\n\nNormality-based scoring is the strongest existing repair for unknown attack settings. CT&T test03 improves substantially in the multi-seed anomaly summary. CT&T test04 improves but remains below a top-tier robustness bar.\n",
        encoding="utf-8",
    )


def e4_sparse(missing: list[str]) -> None:
    d6_path = ROOT / "results/cmf_diagnostics/tables/d6_window_label_dilution.csv"
    d6 = pd.read_csv(d6_path) if d6_path.exists() else pd.DataFrame()
    d6.to_csv(OUT / "e4_sparse_segment.csv", index=False)
    if not d6.empty:
        write_tex(d6, OUT / "e4_sparse_segment.tex")
        fig, ax = plt.subplots(figsize=(6, 3))
        plot = d6[d6["model"].isin(["transformer", "cmf_can"])]
        for model, g in plot.groupby("model"):
            ax.plot(g["bucket"], g["recall"], marker="o", label=clean_model(model))
        ax.set_ylabel("Recall")
        ax.set_ylim(0, 1.0)
        ax.tick_params(axis="x", rotation=25)
        ax.legend(frameon=False)
        ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)
        savefig(fig, "fig_e4_sparse_segment")
    else:
        (OUT / "e4_sparse_segment.tex").write_text("", encoding="utf-8")
    (OUT / "e4_sparse_segment_analysis.md").write_text(
        "# E4 Sparse Segment Evidence\n\nReliable-CMF-CAN implements segment/top-k pooling, but trained results are not available. Existing D6 label-dilution diagnostics are copied as evidence that low attack-frame-ratio windows need segment-level treatment.\n",
        encoding="utf-8",
    )
    missing.append("E4 missing trained Reliable-CMF-CAN segment/top-k checkpoint and segment_scores dump")


def e5_low_fpr(missing: list[str]) -> None:
    low = csv("paper_table_low_fpr_refined.csv")
    opt = pd.concat([csv("optimization_trials_road_1pct.csv"), csv("optimization_trials_ctt_1pct.csv")], ignore_index=True, sort=False)
    out = opt if not opt.empty else low
    out.to_csv(OUT / "e5_low_fpr_objective.csv", index=False)
    write_tex(out, OUT / "e5_low_fpr_objective.tex")
    if not out.empty and "recall_at_fpr_1em03" in out.columns:
        fig, ax = plt.subplots(figsize=(6, 3))
        out.groupby("selection_metric")["recall_at_fpr_1em03"].max().plot(kind="bar", ax=ax, edgecolor="#222")
        ax.set_ylabel("Best Recall@FPR<=1e-3")
        ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)
        savefig(fig, "fig_e5_low_fpr_objective")
    (OUT / "e5_low_fpr_objective_analysis.md").write_text(
        "# E5 Low-FPR Objective\n\nExisting optimization/calibration tables indicate that F1-only selection is not aligned with deployment. Reliable-CMF-CAN has not yet been trained with the low-FPR composite objective.\n",
        encoding="utf-8",
    )
    missing.append("E5 missing Reliable-CMF-CAN training with val_low_fpr_composite")


def e6_e8(missing: list[str]) -> None:
    for name, src in [
        ("e6_multiseed", "ctt_shifted_multiseed_15ep.csv"),
        ("e7_window_sensitivity", ""),
        ("e8_per_attack_failure", "paper_table_per_attack_results.csv"),
    ]:
        df = csv(src) if src else pd.DataFrame()
        df.to_csv(OUT / f"{name}.csv", index=False)
        write_tex(df, OUT / f"{name}.tex")
        if name == "e6_multiseed":
            analysis = "Existing multi-seed shifted CT&T results are available for CMF-CAN, but Reliable-CMF-CAN multi-seed results are missing."
            missing.append("E6 missing Reliable-CMF-CAN multi-seed runs")
        elif name == "e7_window_sensitivity":
            analysis = "Window-size sensitivity requires rebuilding processed windows/features for 50/100/200; no fake result is generated."
            missing.append("E7 missing rebuilt window_size=50/200 features and model runs")
        else:
            analysis = "Per-attack/failure tables exist for current prediction dumps. Reliable-CMF-CAN per-attack results are missing until predictions are exported."
            missing.append("E8 missing Reliable-CMF-CAN prediction dump for per-attack analysis")
        (OUT / f"{name}_analysis.md").write_text(f"# {name}\n\n{analysis}\n", encoding="utf-8")


def success_and_recommendations(missing: list[str]) -> None:
    criteria = [
        ("ROAD/test01 not weaker than Transformer", "partial", "Existing Full CMF-CAN is mixed; simplified variants sometimes help."),
        ("CT&T test02 low-FPR advantage", "partial", "Normality/context evidence helps, but Reliable-CMF-CAN not trained."),
        ("CT&T test03/test04 unknown attack improves", "partial", "Normality improves test03 strongly; test04 remains weak."),
        ("Reliable full model no longer loses to simple variants", "not met", "No trained Reliable-CMF-CAN checkpoint yet."),
        ("Shift-aware context explains unknown vehicle", "partial", "-Ctx evidence supports cause; learned shift-control not yet validated."),
        ("Normality branch improves unknown attack", "met for existing policy", "Anomaly ensemble evidence supports this."),
        ("Segment pooling improves sparse attacks", "not met", "Code exists; no trained result."),
        ("Multi-seed stability", "partial", "Existing anomaly/CMF multi-seed only; Reliable missing."),
        ("Complexity acceptable", "unknown", "Reliable params/latency not profiled."),
        ("Ablation supports mechanisms", "partial", "Existing CMF/anomaly ablations; Reliable ablations missing."),
    ]
    crit = pd.DataFrame(criteria, columns=["criterion", "status", "evidence"])
    crit.to_csv(OUT / "top_tier_success_criteria.csv", index=False)
    gap = ["# Top-Tier Gap Report", "", "Reliable-CMF-CAN is implemented as a research prototype, but the experimental evidence does not yet meet a CCF A/security-four bar.", "", "## Missing or Not Met", ""]
    gap += [f"- {m}" for m in missing]
    gap += ["", "## Assessment", "", "The strongest current direction is Reliable-CMF-CAN as an adaptive low-FPR/open-world system, but the current result set is still closer to CCF B unless P0/P1 Reliable experiments are completed."]
    (OUT / "top_tier_gap_report.md").write_text("\n".join(gap) + "\n", encoding="utf-8")
    (OUT / "recommended_main_model.md").write_text(
        "# Recommended Main Model\n\nDo not replace all results with untrained Reliable-CMF-CAN. Use the original CMF-CAN as the supervised encoder, and define the forward-looking main model as **Reliable-CMF-CAN + normality/adaptive policy** once trained.\n\nCurrent main metrics should be AUPR and Recall@FPR for deployment, with F1 reported but not used as the only headline. The strongest paper line is reliable low-FPR/open-world CAN IDS, not universal F1 superiority.\n",
        encoding="utf-8",
    )
    (OUT / "recommended_experiment_tables.md").write_text(
        "# Recommended Experiment Tables\n\nMain: E1 main results, E2 shift context, E3 normality unknown attack, E5 low-FPR objective, E6 multi-seed once Reliable-CMF-CAN is trained.\n\nAppendix: E4 sparse segment, E7 window sensitivity, E8 per-attack/failure.\n",
        encoding="utf-8",
    )
    (OUT / "recommended_paper_claims.md").write_text(
        "# Recommended Paper Claims\n\n- Cross-modality fusion is useful but unreliable under distribution shift unless modality reliability is controlled.\n- ID-context is domain-sensitive and should be downweighted or masked under unknown-vehicle shift.\n- Benign-normality scoring materially improves unknown-attack detection in selected CT&T settings.\n- Low-FPR deployment requires validation-selected operating points, not fixed thresholds.\n",
        encoding="utf-8",
    )
    (OUT / "unsafe_claims_do_not_write.md").write_text(
        "# Unsafe Claims Do Not Write\n\n- Do not claim Full CMF-CAN consistently outperforms Transformer.\n- Do not claim Reliable-CMF-CAN reaches CCF A/security-four standard before trained multi-seed results exist.\n- Do not claim CT&T test04 is solved.\n- Do not claim segment pooling improves sparse attacks before trained segment/top-k results are available.\n- Do not claim full CrySyS generalization from subset/family-subset evidence.\n",
        encoding="utf-8",
    )
    (OUT / "final_top_tier_upgrade_summary.md").write_text(
        "# Final Top-Tier Upgrade Summary\n\nThis upgrade implements Reliable-CMF-CAN components and generates a truthful evidence package in `results/top_tier_upgrade`. The current evidence supports the research hypothesis direction, but not yet a top-tier acceptance claim. The immediate next step is to train/evaluate Reliable-CMF-CAN and its ablations on ROAD and CT&T test01/test02/test04 with three seeds.\n",
        encoding="utf-8",
    )


def main() -> None:
    setup_style()
    OUT.mkdir(parents=True, exist_ok=True)
    PRED_OUT.mkdir(parents=True, exist_ok=True)
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    missing: list[str] = []
    current_limitations()
    _, pred_missing = copy_prediction_assets()
    missing.extend(pred_missing)
    e1_main(missing)
    e2_context(missing)
    e3_normality(missing)
    e4_sparse(missing)
    e5_low_fpr(missing)
    e6_e8(missing)
    success_and_recommendations(missing)
    (OUT / "missing_report.md").write_text("# Missing Report\n\n" + "\n".join(f"- {m}" for m in missing) + "\n", encoding="utf-8")
    print(f"[reliable_upgrade_assets] missing={len(missing)}")


if __name__ == "__main__":
    main()
