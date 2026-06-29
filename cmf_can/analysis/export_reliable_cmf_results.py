from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score


ROOT = Path(".")
TABLES = ROOT / "results/reliable_cmf/tables"
FIGS = ROOT / "results/reliable_cmf/figures"
PREDS = ROOT / "results/reliable_cmf/predictions"
CMF_TABLES = ROOT / "results/cmf_tables"
OUT = ROOT / "results/reliable_cmf"


def setup() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
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
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def write_tex(df: pd.DataFrame, path: Path) -> None:
    path.write_text(df.to_latex(index=False, escape=True, na_rep="NA", float_format=lambda x: f"{x:.4f}"), encoding="utf-8")


def savefig(fig: plt.Figure, name: str) -> None:
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIGS / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def model_label(m: str) -> str:
    return {
        "transformer": "Transformer",
        "frame_only": "Frame-only",
        "stats_only": "Stats-only",
        "concat_fusion": "Concat-Fusion",
        "cmf_can": "CMF-CAN",
        "wo_context": "CMF-CAN -Ctx",
        "wo_stats": "CMF-CAN -Win",
        "reliable_cmf_can": "Reliable-CMF-CAN",
    }.get(m, m)


def append_metric_row(rows: list[dict], rec: dict, source: str, setting: str | None = None) -> None:
    rows.append(
        {
            "dataset": setting or rec.get("dataset"),
            "model": rec.get("model"),
            "Model": model_label(str(rec.get("model"))),
            "seed": rec.get("seed", 42),
            "precision": rec.get("precision", np.nan),
            "recall": rec.get("recall", np.nan),
            "f1": rec.get("f1", np.nan),
            "macro_f1": rec.get("macro_f1", np.nan),
            "auroc": rec.get("auroc", np.nan),
            "aupr": rec.get("aupr", np.nan),
            "fpr": rec.get("fpr", np.nan),
            "fnr": rec.get("fnr", np.nan),
            "recall_at_fpr_1em04": rec.get("recall_at_fpr_1em04", np.nan),
            "recall_at_fpr_5em04": rec.get("recall_at_fpr_5em04", np.nan),
            "recall_at_fpr_1em03": rec.get("recall_at_fpr_1em03", np.nan),
            "selection_metric": rec.get("selection_metric", "f1"),
            "source": source,
        }
    )


def main_table() -> pd.DataFrame:
    rows: list[dict] = []
    reliable = read(CMF_TABLES / "reliable_cmf_main_raw.csv")
    road = read(CMF_TABLES / "road_main_20ep.csv")
    ctt = read(CMF_TABLES / "ctt_generalization_15ep.csv")
    ctt_ab = read(CMF_TABLES / "ctt_unknown_ablation.csv")
    for model in ["transformer", "frame_only", "stats_only", "concat_fusion", "cmf_can"]:
        hit = road[road["model"].eq(model)]
        if not hit.empty:
            append_metric_row(rows, hit.iloc[0].to_dict(), "road_main_20ep.csv", "road")
    for ds in ["ctt_test01", "ctt_test02", "ctt_test03", "ctt_test04"]:
        for model in ["transformer", "frame_only", "stats_only", "concat_fusion", "cmf_can", "wo_context"]:
            source = ctt_ab if model in {"frame_only", "stats_only", "wo_context"} else ctt
            hit = source[(source["dataset"].eq(ds)) & (source["model"].eq(model))]
            if not hit.empty:
                append_metric_row(rows, hit.iloc[0].to_dict(), source="ctt_unknown_ablation.csv" if source is ctt_ab else "ctt_generalization_15ep.csv", setting=ds)
    for _, rec in reliable.iterrows():
        append_metric_row(rows, rec.to_dict(), "reliable_cmf_main_raw.csv")
    out = pd.DataFrame(rows)
    wanted = ["road", "ctt_test01", "ctt_test02", "ctt_test03", "ctt_test04"]
    out["dataset"] = pd.Categorical(out["dataset"], categories=wanted, ordered=True)
    out = out.sort_values(["dataset", "Model"]).reset_index(drop=True)
    out.to_csv(TABLES / "main_reliable_cmf_single_seed.csv", index=False)
    write_tex(out, TABLES / "main_reliable_cmf_single_seed.tex")
    plot = out[out["Model"].isin(["Transformer", "Frame-only", "Stats-only", "Concat-Fusion", "CMF-CAN", "CMF-CAN -Ctx", "Reliable-CMF-CAN"])]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    datasets = [d for d in wanted if d in set(plot["dataset"].astype(str))]
    models = list(dict.fromkeys(plot["Model"]))
    x = np.arange(len(datasets))
    width = min(0.8 / max(len(models), 1), 0.12)
    for i, model in enumerate(models):
        vals = []
        for ds in datasets:
            hit = plot[(plot["dataset"].astype(str).eq(ds)) & (plot["Model"].eq(model))]
            vals.append(float(hit["f1"].iloc[0]) if not hit.empty and pd.notna(hit["f1"].iloc[0]) else np.nan)
        ax.bar(x + (i - (len(models) - 1) / 2) * width, vals, width, label=model, edgecolor="#222", hatch="//" if model == "Reliable-CMF-CAN" else None)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=20, ha="right")
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, ncol=3)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)
    savefig(fig, "main_reliable_cmf_single_seed")
    return out


def ablation_table(main: pd.DataFrame) -> pd.DataFrame:
    keep = main[main["Model"].isin(["Frame-only", "Stats-only", "CMF-CAN", "CMF-CAN -Ctx", "Reliable-CMF-CAN"])].copy()
    missing_models = [
        "Reliable-CMF-CAN w/o reliability gate",
        "Reliable-CMF-CAN w/o shift control",
        "Reliable-CMF-CAN w/o normality",
        "Reliable-CMF-CAN w/o segment pooling",
    ]
    rows = keep.to_dict("records")
    for ds in ["road", "ctt_test01", "ctt_test02", "ctt_test04"]:
        for model in missing_models:
            rows.append({"dataset": ds, "Model": model, "model": model, "f1": np.nan, "macro_f1": np.nan, "aupr": np.nan, "auroc": np.nan, "source": "not run"})
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "ablation_reliable_cmf.csv", index=False)
    write_tex(out, TABLES / "ablation_reliable_cmf.tex")
    plot = out[out["dataset"].astype(str).isin(["road", "ctt_test02"]) & out["Model"].isin(["Frame-only", "Stats-only", "CMF-CAN", "CMF-CAN -Ctx", "Reliable-CMF-CAN"])]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    for model, g in plot.groupby("Model"):
        ax.bar([f"{r.dataset}\n{model}" for r in g.itertuples()], g["f1"], label=model, edgecolor="#222")
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)
    savefig(fig, "ablation_reliable_cmf")
    return out


def low_fpr_table() -> pd.DataFrame:
    raw = read(CMF_TABLES / "reliable_cmf_lowfpr_raw.csv")
    if raw.empty:
        raw = pd.DataFrame()
    raw.to_csv(TABLES / "low_fpr_objective.csv", index=False)
    write_tex(raw, TABLES / "low_fpr_objective.tex")
    if not raw.empty:
        fig, ax = plt.subplots(figsize=(5.6, 3.0))
        for ds, g in raw.groupby("dataset"):
            ax.plot(g["selection_metric"], g["recall_at_fpr_1em03"], marker="o", label=ds)
        ax.set_ylabel("Recall@FPR<=1e-3")
        ax.tick_params(axis="x", rotation=20)
        ax.legend(frameon=False)
        ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)
        savefig(fig, "low_fpr_objective")
    return raw


def per_attack() -> pd.DataFrame:
    rows = []
    for path in PREDS.glob("*reliable_cmf_can_predictions.csv"):
        df = pd.read_csv(path)
        dataset = str(df["dataset"].iloc[0])
        model = str(df["model"].iloc[0])
        for attack, g in df.groupby("attack_type"):
            y = g["label"].astype(int)
            pred = g["prediction"].astype(int)
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "attack_type": attack,
                    "support": int(len(g)),
                    "precision": precision_score(y, pred, zero_division=0),
                    "recall": recall_score(y, pred, zero_division=0),
                    "f1": f1_score(y, pred, zero_division=0),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "per_attack_reliable_cmf.csv", index=False)
    write_tex(out, TABLES / "per_attack_reliable_cmf.tex")
    if not out.empty:
        plot = out[out["attack_type"].ne("normal")].head(20)
        fig, ax = plt.subplots(figsize=(7, 3.2))
        ax.bar(plot["dataset"].astype(str) + "\n" + plot["attack_type"].astype(str), plot["recall"], edgecolor="#222")
        ax.set_ylabel("Recall")
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)
        savefig(fig, "per_attack_reliable_cmf")
    return out


def segment_evidence_files() -> list[Path]:
    written: list[Path] = []
    for path in PREDS.glob("*reliable_cmf_can_gate_weights.csv"):
        df = pd.read_csv(path)
        cols = {
            "sample_id": df.get("sample_id"),
            "dataset": df.get("dataset"),
            "setting": df.get("setting", df.get("dataset")),
            "model": df.get("model"),
            "segment_scores": "NA",
            "topk_segment_indices": "NA",
            "topk_score": df.get("topk_score", np.nan),
            "label": df.get("label"),
            "attack_type": df.get("attack_type", "NA"),
        }
        out = pd.DataFrame(cols)
        target = path.with_name(path.name.replace("_gate_weights.csv", "_segment_scores.csv"))
        out.to_csv(target, index=False)
        written.append(target)
    return written


def write_findings(main: pd.DataFrame, low: pd.DataFrame, per: pd.DataFrame) -> None:
    missing = [
        "CT&T test01/test03/test04 Reliable-CMF-CAN full 50ep runs not completed in this turn.",
        "Reliable-CMF-CAN ablation variants were registered but not trained: no_reliability, no_shift, no_normality, no_segment.",
        "Segment evidence CSVs contain real topk_score, but full per-segment score vectors and top-k indices were not exported by the trainer.",
        "Three-seed CT&T shifted validation was not run because single-seed Reliable-CMF-CAN did not beat existing alternatives.",
        "Event-level metrics for Reliable-CMF-CAN were not recomputed in this turn.",
    ]
    (OUT / "missing_or_failed_runs.md").write_text("# Missing or Failed Runs\n\n" + "\n".join(f"- {m}" for m in missing) + "\n", encoding="utf-8")

    road_rel = main[(main["dataset"].astype(str).eq("road")) & (main["Model"].eq("Reliable-CMF-CAN"))]
    ctt2_rel = main[(main["dataset"].astype(str).eq("ctt_test02")) & (main["Model"].eq("Reliable-CMF-CAN"))]
    lines = [
        "# Ablation Findings",
        "",
        "1. Reliability Gate 是否比普通 gate 更好：未能证明。Reliable-CMF-CAN Full 在已完成的 ROAD/test02 上没有超过旧 CMF-CAN 或最强简化变体。",
        "2. Shift-aware Context 是否改善 test02/test04：未能证明。test02 Reliable Full F1 仍为 0.1600，明显低于已存在的 -Ctx 单 seed 结果。",
        "3. Normality Branch 是否改善 unknown attack：Reliable 模型训练未包含可学习 normality 分支；现有 anomaly policy 仍是 unknown attack 的更强证据。",
        "4. Segment Pooling 是否改善 ROAD/per-attack：未能证明。ROAD Reliable F1 低于 Transformer 和旧 CMF-CAN。",
        "5. Full 是否仍输给简化模型：是。ROAD 上低于 Transformer/旧 CMF-CAN；CT&T test02 上低于 -Ctx 和 anomaly policy。",
        "6. 当前主模型建议：不要切换到 Reliable-CMF-CAN Full；ROAD 仍用 Transformer/旧 CMF-CAN 做强 baseline，CT&T test02 用 -Ctx 或 anomaly/calibrated policy。",
    ]
    (OUT / "ablation_findings.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    (OUT / "low_fpr_objective_findings.md").write_text(
        "# Low-FPR Objective Findings\n\n"
        "- ROAD: Recall@FPR<=1e-3 selection raises thresholded F1 from 0.7877 to 0.8137, but it still does not beat Transformer F1 0.8279.\n"
        "- CT&T test02: Recall@FPR<=1e-3 remains 0.4540 for Reliable-CMF-CAN, below old CMF-CAN/anomaly-policy evidence.\n"
        "- Main metric should include Recall@FPR/AUPR, but changing threshold alone does not make Reliable-CMF-CAN the new main model.\n",
        encoding="utf-8",
    )
    (OUT / "per_attack_findings.md").write_text(
        "# Per-Attack Findings\n\nReliable-CMF-CAN prediction dumps exist only for ROAD and CT&T test02 in this run. Per-attack conclusions for test03/test04 cannot be made without prediction dumps.\n",
        encoding="utf-8",
    )
    selection_rows = [
        {"setting": "ROAD", "recommended_model": "Transformer or old CMF-CAN depending metric", "reason": "Reliable-CMF-CAN F1=0.7877, lower than Transformer 0.8279 and old CMF-CAN 0.7894; Recall@FPR thresholding improves but not enough.", "main_metric": "F1 + Recall@FPR"},
        {"setting": "CT&T test02", "recommended_model": "CMF-CAN -Ctx / anomaly calibrated policy", "reason": "Reliable-CMF-CAN F1=0.1600 and Recall@FPR<=1e-3=0.4540; -Ctx/anomaly evidence is stronger.", "main_metric": "Recall@FPR + AUPR"},
        {"setting": "CT&T test03/test04", "recommended_model": "Anomaly/normality policy until Reliable is trained", "reason": "No Reliable-CMF-CAN 50ep result in this run; existing anomaly policy is the only strong unknown-attack evidence.", "main_metric": "AUPR + event recall under false alarm budget"},
    ]
    sel = pd.DataFrame(selection_rows)
    sel.to_csv(OUT / "final_model_selection.csv", index=False)
    (OUT / "final_model_selection.md").write_text(
        "# Final Model Selection\n\n"
        "Reliable-CMF-CAN Full should **not** replace CMF-CAN as the main model based on the completed runs.\n\n"
        + "\n".join(f"- **{r['setting']}**: {r['recommended_model']} — {r['reason']}" for r in selection_rows)
        + "\n\nCurrent result is CCF B-style evidence at best, not sufficient for CCF A / Security Four.\n",
        encoding="utf-8",
    )
    (OUT / "final_reliable_cmf_report.md").write_text(
        "# Final Reliable-CMF-CAN Report\n\n"
        "## What was implemented\nReliable-CMF-CAN was already registered and smoke-tested; this run trained it on ROAD and CT&T test02 for 50 epochs.\n\n"
        "## What was trained\n- ROAD / reliable_cmf_can / seed 42 / 50 epochs\n- CT&T test02 / reliable_cmf_can / seed 42 / 50 epochs\n\n"
        "## What improved\n- ROAD F1 improves under Recall@FPR threshold selection from 0.7877 to 0.8137, but not enough to beat Transformer.\n\n"
        "## What did not improve\n- ROAD Reliable Full does not beat Transformer or old CMF-CAN.\n- CT&T test02 Reliable Full remains at F1 0.1600 and does not beat -Ctx/anomaly policy.\n\n"
        "## Best model per setting\nSee `final_model_selection.csv`.\n\n"
        "## Recommended paper claim\nDo not claim Reliable-CMF-CAN is the new main model yet. Claim it is an implemented prototype whose first training results show the need for simpler context-controlled or anomaly-calibrated variants.\n\n"
        "## Remaining gap to CCF A / Security Four\nNeed trained ablations, test04 results, multi-seed stability, event-level metrics, and a working normality branch integrated into the Reliable score.\n",
        encoding="utf-8",
    )


def main() -> None:
    setup()
    main = main_table()
    ablation_table(main)
    low = low_fpr_table()
    per = per_attack()
    segment_evidence_files()
    write_findings(main, low, per)
    print("[export_reliable_cmf_results] done")


if __name__ == "__main__":
    main()
