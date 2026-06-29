from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_curve, precision_score, recall_score


ROOT = Path(".")
OUT = ROOT / "results/transformer_rescue"
TABLES = OUT / "tables"
FIGS = OUT / "figures"
CMF_TABLES = ROOT / "results/cmf_tables"
PREDS = OUT / "predictions"

TRANSFORMER_F1 = 0.8279109589041096
OLD_CMF_F1 = 0.7893805309734513
OLD_CMF_AUPR = 0.805957074624345


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


def model_label(model: str) -> str:
    return {
        "transformer": "Transformer",
        "cmf_can": "CMF-CAN",
        "can_transformer_plus_sameid": "CAN-Transformer+ same-ID",
        "tfscan_gate": "TFS-CAN gate",
    }.get(model, model)


def road_table() -> pd.DataFrame:
    rows: list[dict] = []
    road = read(CMF_TABLES / "road_main_20ep.csv")
    raw = read(CMF_TABLES / "transformer_rescue_raw.csv")
    for model in ["transformer", "cmf_can"]:
        hit = road[road["model"].eq(model)]
        if not hit.empty:
            rec = hit.iloc[0].to_dict()
            rec["source"] = "road_main_20ep.csv"
            rows.append(rec)
    for _, rec in raw[raw["dataset"].eq("road")].iterrows():
        item = rec.to_dict()
        item["source"] = "transformer_rescue_raw.csv"
        rows.append(item)
    out = pd.DataFrame(rows)
    out["Model"] = out["model"].map(model_label)
    out["exceeds_transformer_f1"] = out["f1"] > TRANSFORMER_F1
    out["exceeds_old_cmf_f1"] = out["f1"] > OLD_CMF_F1
    out["exceeds_old_cmf_aupr"] = out["aupr"] > OLD_CMF_AUPR
    cols = [
        "dataset",
        "Model",
        "model",
        "precision",
        "recall",
        "f1",
        "macro_f1",
        "auroc",
        "aupr",
        "fpr",
        "fnr",
        "recall_at_fpr_1em04",
        "recall_at_fpr_5em04",
        "recall_at_fpr_1em03",
        "exceeds_transformer_f1",
        "exceeds_old_cmf_f1",
        "exceeds_old_cmf_aupr",
        "selection_metric",
        "source",
    ]
    out = out[[c for c in cols if c in out.columns]]
    out.to_csv(TABLES / "road_transformer_rescue.csv", index=False)
    write_tex(out, TABLES / "road_transformer_rescue.tex")

    plot = out[out["Model"].isin(["Transformer", "CMF-CAN", "CAN-Transformer+ same-ID", "TFS-CAN gate"])]
    fig, ax = plt.subplots(figsize=(5.8, 3.2))
    x = np.arange(len(plot))
    colors = ["#4C78A8", "#72B7B2", "#F58518", "#54A24B"]
    ax.bar(x, plot["f1"], color=colors[: len(plot)], edgecolor="#222")
    ax.axhline(TRANSFORMER_F1, color="#111", linestyle="--", linewidth=1.0, label="Transformer F1")
    ax.set_xticks(x)
    ax.set_xticklabels(plot["Model"], rotation=20, ha="right")
    ax.set_ylabel("F1")
    ax.set_ylim(0.75, 0.85)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)
    ax.legend(frameon=False)
    savefig(fig, "road_transformer_rescue")
    return out


def ctt_table() -> pd.DataFrame:
    ctt = read(CMF_TABLES / "ctt_generalization_15ep.csv")
    ab = read(CMF_TABLES / "ctt_unknown_ablation.csv")
    rows = []
    for ds in ["ctt_test01", "ctt_test02"]:
        for source, models in [(ctt, ["transformer", "cmf_can"]), (ab, ["wo_context", "stats_only", "frame_only"])]:
            for model in models:
                hit = source[(source["dataset"].eq(ds)) & (source["model"].eq(model))]
                if not hit.empty:
                    item = hit.iloc[0].to_dict()
                    item["source"] = "existing_evidence"
                    rows.append(item)
        rows.append({"dataset": ds, "model": "can_transformer_plus_sameid", "Model": "CAN-Transformer+ same-ID", "source": "not run after ROAD did not exceed Transformer"})
        rows.append({"dataset": ds, "model": "tfscan_gate", "Model": "TFS-CAN gate", "source": "not run after ROAD did not exceed Transformer"})
    out = pd.DataFrame(rows)
    if "Model" not in out:
        out["Model"] = out["model"].map(model_label)
    else:
        out["Model"] = out["Model"].fillna(out["model"].map(model_label))
    out.to_csv(TABLES / "ctt_transformer_rescue.csv", index=False)
    write_tex(out, TABLES / "ctt_transformer_rescue.tex")

    plot = out[pd.to_numeric(out.get("f1"), errors="coerce").notna()].copy()
    if not plot.empty:
        fig, ax = plt.subplots(figsize=(6.2, 3.2))
        labels = plot["dataset"].astype(str) + "\n" + plot["Model"].astype(str)
        ax.bar(np.arange(len(plot)), plot["f1"].astype(float), edgecolor="#222")
        ax.set_xticks(np.arange(len(plot)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("F1")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)
        savefig(fig, "ctt_transformer_rescue")
    return out


def threshold_calibration() -> pd.DataFrame:
    rows = []
    for path in PREDS.glob("road_*_predictions.csv"):
        df = pd.read_csv(path)
        y = df["label"].astype(int).to_numpy()
        s = df["score"].astype(float).to_numpy()
        model = str(df["model"].iloc[0])
        formal_t = float(df["threshold"].iloc[0]) if "threshold" in df else np.nan
        prec_curve, rec_curve, thr_curve = precision_recall_curve(y, s)
        f1_curve = 2 * prec_curve[:-1] * rec_curve[:-1] / np.maximum(prec_curve[:-1] + rec_curve[:-1], 1e-12)
        best_t = float(thr_curve[int(np.nanargmax(f1_curve))]) if len(thr_curve) else 0.5
        thresholds = {
            "default_0.5": 0.5,
            "formal_validation_selected": formal_t,
            "best_test_upper_bound": best_t,
        }
        for name, t in thresholds.items():
            pred = (s >= t).astype(int)
            rows.append(
                {
                    "dataset": "road",
                    "model": model,
                    "threshold_policy": name,
                    "threshold": t,
                    "precision": precision_score(y, pred, zero_division=0),
                    "recall": recall_score(y, pred, zero_division=0),
                    "f1": f1_score(y, pred, zero_division=0),
                    "formal_result": name != "best_test_upper_bound",
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "threshold_calibration.csv", index=False)
    write_tex(out, TABLES / "threshold_calibration.tex")
    (OUT / "threshold_calibration_analysis.md").write_text(
        "# Threshold Calibration Analysis\n\n"
        "Only test prediction dumps are available in this run. The formal threshold is the validation-selected threshold saved by training; best-test threshold is reported only as an upper bound and must not be used as the main result.\n\n"
        "- CAN-Transformer+ same-ID improves over old CMF-CAN F1 and AUPR, but does not exceed Transformer F1.\n"
        "- TFS-CAN gate gets closer under FPR-constrained thresholds, but still does not beat Transformer F1.\n",
        encoding="utf-8",
    )
    return out


def write_reports(road: pd.DataFrame) -> None:
    best = road.sort_values("f1", ascending=False).iloc[0]
    rescue = road[road["model"].isin(["can_transformer_plus_sameid", "tfscan_gate"])]
    missing = [
        "P0 ROAD full matrix not completed: basic, time-bias, same-ID+time-bias, attention-pool, top-k, TFS concat, TFS stats-token attention.",
        "CT&T test01/test02 transformer-rescue candidates were not trained because ROAD did not exceed Transformer F1.",
        "Focal loss + val_aupr and weighted CE + val_recall_at_fpr_1e-3 training were not run for rescue candidates.",
        "Validation prediction dumps were not exported, so precision-constrained validation threshold is unavailable.",
        "True Mamba/SSM baseline was not run; no mamba/causal-conv/selective-scan dependency is available in the current environment.",
    ]
    (OUT / "missing_or_failed_runs.md").write_text("# Missing or Failed Runs\n\n" + "\n".join(f"- {m}" for m in missing) + "\n", encoding="utf-8")
    (OUT / "road_failure_analysis.md").write_text(
        "# ROAD Failure Analysis\n\n"
        f"Best completed rescue model by F1 is **{best['Model']}** with F1={best['f1']:.4f}. "
        f"The Transformer baseline remains higher at F1={TRANSFORMER_F1:.4f}.\n\n"
        "1. Transformer 是否接近当前特征和数据协议上限：当前证据支持这个判断。CAN-specific same-ID features lifted AUPR slightly, but did not pass Transformer F1.\n"
        "2. 无效模块：TFS residual gate did not help ROAD F1 or AUPR; window stats appear to dilute rather than improve the strong sequence signal.\n"
        "3. 是否需要改 window size：需要。当前 window=100 可能让 sparse attack recall 受限；应优先测试 50/100/200。\n"
        "4. 是否需要 Mamba/SSM：可以作为下一步，但必须使用真实 Mamba/SSM implementation，不应伪造 Mamba baseline。\n"
        "5. 是否需要 per-attack 专门优化：需要，尤其是 sparse/fuzzing/short-span attacks。\n"
        "6. 是否停止当前方案：应停止继续堆 fusion；保留 CAN-Transformer+ same-ID 作为 AUPR 改进证据，但不要称其稳定超过 Transformer。\n",
        encoding="utf-8",
    )
    decision_rows = [
        {
            "question": "abandon_reliable_cmf_full",
            "decision": "yes",
            "evidence": "Reliable-CMF-CAN Full failed on ROAD and CT&T test02.",
        },
        {
            "question": "choose_can_transformer_plus",
            "decision": "not as main by F1; candidate by AUPR",
            "evidence": "ROAD F1=0.8229 < Transformer 0.8279; AUPR=0.8078 > old CMF-CAN 0.8060.",
        },
        {
            "question": "choose_tfscan",
            "decision": "no",
            "evidence": "ROAD TFS-CAN gate F1=0.8101 and AUPR=0.7840, worse than CAN-Transformer+ same-ID.",
        },
        {
            "question": "need_mamba",
            "decision": "maybe",
            "evidence": "Transformer remains the F1 ceiling among completed models; real SSM/Mamba is a valid next baseline if dependency is installed.",
        },
        {
            "question": "paper_possibility",
            "decision": "possible only with cautious framing",
            "evidence": "Current rescue does not create a new dominant model; a paper must focus on threshold/low-FPR/AUPR or systematic evidence, not SOTA F1.",
        },
    ]
    pd.DataFrame(decision_rows).to_csv(OUT / "final_model_decision.csv", index=False)
    (OUT / "final_model_decision.md").write_text(
        "# Final Model Decision\n\n"
        "- Reliable-CMF-CAN Full should be abandoned as the main model.\n"
        "- CAN-Transformer+ same-ID is the best completed rescue candidate, but it does **not** truly exceed Transformer F1.\n"
        "- TFS-CAN gate should not be selected as main model.\n"
        "- If the target is a stronger paper, the next necessary experiment is not more fusion; it is window-size/per-attack optimization or a real SSM/Mamba sequence backbone.\n"
        "- Main metric should remain F1 for ROAD comparability, with AUPR/Recall@FPR reported as secondary deployment metrics.\n",
        encoding="utf-8",
    )
    (OUT / "mamba_baseline_analysis.md").write_text(
        "# Mamba Baseline Analysis\n\n"
        "Mamba was not run in this turn. The current environment does not expose a verified Mamba/selective-scan implementation. A placeholder would be misleading, so no fake Mamba result was generated.\n",
        encoding="utf-8",
    )


def main() -> None:
    setup()
    road = road_table()
    ctt_table()
    threshold_calibration()
    write_reports(road)
    print("[export_transformer_rescue] done")


if __name__ == "__main__":
    main()
