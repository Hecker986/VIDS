from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from sklearn.metrics import auc, precision_recall_curve, roc_curve
except Exception:  # pragma: no cover
    auc = precision_recall_curve = roc_curve = None


TABLE_DIR = Path("results/cmf_tables")
FIG_DIR = Path("results/cmf_figures")
PRED_DIR = Path("results/cmf_predictions")
DATA_DIR = Path("data/processed")

MODEL_ORDER = ["transformer", "concat_fusion", "cmf_can"]
MODEL_LABEL = {"transformer": "Transformer", "concat_fusion": "Concat-Fusion", "cmf_can": "CMF-CAN"}
LABEL_MODEL = {v: k for k, v in MODEL_LABEL.items()}
COLORS = {"Transformer": "#4C78A8", "Concat-Fusion": "#F2BE3E", "CMF-CAN": "#E45756"}
HATCH = {"Transformer": "///", "Concat-Fusion": "\\\\\\", "CMF-CAN": ""}
MARKER = {"Transformer": "o", "Concat-Fusion": "s", "CMF-CAN": "^"}
LINE = {"Transformer": "-", "Concat-Fusion": "--", "CMF-CAN": "-."}


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "legend.fontsize": 7.8,
            "figure.titlesize": 11,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": "#E5E7EB",
            "grid.linewidth": 0.7,
            "lines.linewidth": 1.55,
            "patch.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def savefig(fig: mpl.figure.Figure, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(FIG_DIR / f"{stem}.{ext}", dpi=300 if ext == "png" else None, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def fmt(x, digits: int = 4) -> str:
    if x is None or pd.isna(x):
        return "NA"
    return f"{float(x):.{digits}f}"


def tex(df: pd.DataFrame, path: Path) -> None:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].map(lambda x: fmt(x))
    out.to_latex(path, index=False, escape=False, na_rep="NA")


def model_bar(ax: mpl.axes.Axes, df: pd.DataFrame, xcol: str, ycol: str, order: list[str]) -> None:
    width = 0.24
    x = np.arange(len(order))
    for i, model in enumerate(["Transformer", "Concat-Fusion", "CMF-CAN"]):
        vals = df[df["Model"].eq(model)].set_index(xcol).reindex(order)[ycol]
        ax.bar(
            x + (i - 1) * width,
            vals,
            width,
            label=model,
            color=COLORS[model],
            edgecolor="#222222",
            hatch=HATCH[model],
            linewidth=1.25 if model == "CMF-CAN" else 0.8,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=18, ha="right")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y")


def source_metric(row: pd.Series) -> str:
    return "macro_f1" if "macro_f1" in row and pd.notna(row["macro_f1"]) else "f1"


def fig_architecture() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.set_axis_off()
    boxes = {
        "CAN Window": (0.05, 0.62, 0.16, 0.18),
        "Frame-level\nsequence\n[B, L, d]": (0.28, 0.76, 0.18, 0.17),
        "Window-level\nstatistics\n[B, d]": (0.28, 0.52, 0.18, 0.17),
        "ID-level\ncontext\n[B, d]": (0.28, 0.28, 0.18, 0.17),
        "Modality\nencoders": (0.54, 0.54, 0.16, 0.18),
        "Cross-modality\nfusion\n[B, 3, d]": (0.76, 0.66, 0.18, 0.16),
        "Gated\nfusion": (0.76, 0.42, 0.18, 0.15),
        "Classifier /\ndetection score": (0.76, 0.20, 0.18, 0.15),
    }
    for text, (x, y, w, h) in boxes.items():
        fc = "#F8FAFC" if "CMF" not in text else "#FFF7ED"
        lw = 1.4 if text in {"Cross-modality\nfusion\n[B, 3, d]", "Gated\nfusion"} else 0.9
        ax.add_patch(mpl.patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015,rounding_size=0.012", fc=fc, ec="#222222", lw=lw))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color="#111827")
    arrows = [
        ("CAN Window", "Frame-level\nsequence\n[B, L, d]"),
        ("CAN Window", "Window-level\nstatistics\n[B, d]"),
        ("CAN Window", "ID-level\ncontext\n[B, d]"),
        ("Frame-level\nsequence\n[B, L, d]", "Modality\nencoders"),
        ("Window-level\nstatistics\n[B, d]", "Modality\nencoders"),
        ("ID-level\ncontext\n[B, d]", "Modality\nencoders"),
        ("Modality\nencoders", "Cross-modality\nfusion\n[B, 3, d]"),
        ("Cross-modality\nfusion\n[B, 3, d]", "Gated\nfusion"),
        ("Gated\nfusion", "Classifier /\ndetection score"),
    ]
    for a, b in arrows:
        x1, y1, w1, h1 = boxes[a]
        x2, y2, w2, h2 = boxes[b]
        ax.annotate("", xy=(x2, y2 + h2 / 2), xytext=(x1 + w1, y1 + h1 / 2), arrowprops=dict(arrowstyle="->", lw=1.0, color="#374151"))
    ax.text(0.05, 0.94, "CMF-CAN Architecture", fontsize=12, fontweight="bold", ha="left")
    ax.text(0.05, 0.04, "Core model only: packet/frame, window statistics and ID-context modalities are fused before binary detection.", fontsize=8, color="#4B5563")
    savefig(fig, "paper_fig1_architecture_refined")


def main_results() -> pd.DataFrame:
    rows: list[dict] = []
    road = pd.read_csv(TABLE_DIR / "road_main_20ep.csv")
    ctt = pd.read_csv(TABLE_DIR / "ctt_generalization_15ep.csv")
    cry_path = TABLE_DIR / "crysys_family_mod_3model_3seed_mean_std.csv"
    for _, r in road[road["model"].isin(MODEL_ORDER)].iterrows():
        rows.append({"Dataset/Setting": "ROAD", "Model": MODEL_LABEL[r["model"]], "Metric": source_metric(r), "Score": r[source_metric(r)], **r.to_dict()})
    ctt_map = {"ctt_test01": "CT&T KV-KA", "ctt_test02": "CT&T UV-KA", "ctt_test03": "CT&T KV-UA", "ctt_test04": "CT&T UV-UA"}
    for _, r in ctt[ctt["model"].isin(MODEL_ORDER)].iterrows():
        rows.append({"Dataset/Setting": ctt_map[r["dataset"]], "Model": MODEL_LABEL[r["model"]], "Metric": source_metric(r), "Score": r[source_metric(r)], **r.to_dict()})
    if cry_path.exists():
        cry = pd.read_csv(cry_path)
        piv = cry[cry["model"].isin(MODEL_ORDER)].pivot_table(index="model", columns="metric", values="mean", aggfunc="first")
        for model in MODEL_ORDER:
            if model in piv.index:
                score = piv.loc[model, "macro_f1"] if "macro_f1" in piv.columns else piv.loc[model, "f1"]
                rows.append({"Dataset/Setting": "CrySyS-subset", "Model": MODEL_LABEL[model], "Metric": "macro_f1", "Score": score})
    out = pd.DataFrame(rows)
    out[["Dataset/Setting", "Model", "Metric", "Score"]].to_csv(TABLE_DIR / "paper_table_main_multidataset_refined.csv", index=False)
    tex(out[["Dataset/Setting", "Model", "Metric", "Score"]], TABLE_DIR / "paper_table_main_multidataset_refined.tex")
    order = ["ROAD", "CT&T KV-KA", "CT&T UV-KA", "CT&T KV-UA", "CT&T UV-UA", "CrySyS-subset"]
    order = [x for x in order if x in set(out["Dataset/Setting"])]
    fig, ax = plt.subplots(figsize=(7.1, 3.0))
    model_bar(ax, out, "Dataset/Setting", "Score", order)
    ax.set_ylabel("Macro-F1 (fallback: F1)")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.26))
    savefig(fig, "paper_fig2_main_multidataset_refined")
    return out


def few_label(dataset: str, src: str, fig_stem: str, out_stem: str) -> pd.DataFrame:
    df = pd.read_csv(TABLE_DIR / src)
    if "dataset" in df.columns:
        df = df[df["dataset"].eq(dataset)]
    df = df[df["model"].isin(MODEL_ORDER)].copy()
    df["Model"] = df["model"].map(MODEL_LABEL)
    cols = ["label_ratio", "Model", "f1_mean", "f1_std", "precision_mean", "recall_mean", "auroc_mean", "aupr_mean"]
    out = df[[c for c in cols if c in df.columns]].sort_values(["label_ratio", "Model"])
    out.to_csv(TABLE_DIR / f"{out_stem}.csv", index=False)
    tex(out, TABLE_DIR / f"{out_stem}.tex")
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    for model in ["Transformer", "Concat-Fusion", "CMF-CAN"]:
        sub = out[out["Model"].eq(model)].sort_values("label_ratio")
        if sub.empty:
            continue
        x = sub["label_ratio"].to_numpy(float) * 100
        y = sub["f1_mean"].to_numpy(float)
        ax.plot(x, y, label=model, color=COLORS[model], marker=MARKER[model], linestyle=LINE[model], markeredgecolor="#222222")
        if "f1_std" in sub and sub["f1_std"].notna().any():
            s = sub["f1_std"].fillna(0).to_numpy(float)
            ax.fill_between(x, np.clip(y - s, 0, 1), np.clip(y + s, 0, 1), color=COLORS[model], alpha=0.16, lw=0)
    ax.set_xscale("log")
    ax.set_xticks([1, 5, 10, 20, 100])
    ax.set_xticklabels(["1%", "5%", "10%", "20%", "100%"])
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Labeled ratio")
    ax.set_ylabel("F1")
    ax.grid(axis="both", which="both")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.24))
    savefig(fig, fig_stem)
    return out


def ctt_generalization() -> pd.DataFrame:
    df = pd.read_csv(TABLE_DIR / "ctt_generalization_15ep.csv")
    df = df[df["model"].isin(MODEL_ORDER)].copy()
    mapping = {"ctt_test01": "KV-KA", "ctt_test02": "UV-KA", "ctt_test03": "KV-UA", "ctt_test04": "UV-UA"}
    df["Setting"] = df["dataset"].map(mapping)
    df["Model"] = df["model"].map(MODEL_LABEL)
    rows = []
    for setting, g in df.groupby("Setting", sort=False):
        base_t = g[g["model"].eq("transformer")].iloc[0] if (g["model"].eq("transformer")).any() else None
        base_c = g[g["model"].eq("concat_fusion")].iloc[0] if (g["model"].eq("concat_fusion")).any() else None
        for _, r in g.iterrows():
            rows.append(
                {
                    "Setting": setting,
                    "Model": r["Model"],
                    "F1": r["f1"],
                    "Macro-F1": r["macro_f1"],
                    "AUROC": r["auroc"],
                    "AUPR": r["aupr"],
                    "relative_improvement_vs_transformer": np.nan if base_t is None or base_t["macro_f1"] == 0 else (r["macro_f1"] - base_t["macro_f1"]) / base_t["macro_f1"],
                    "relative_improvement_vs_concat": np.nan if base_c is None or base_c["macro_f1"] == 0 else (r["macro_f1"] - base_c["macro_f1"]) / base_c["macro_f1"],
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "paper_table_ctt_generalization_refined.csv", index=False)
    tex(out, TABLE_DIR / "paper_table_ctt_generalization_refined.tex")
    fig, ax = plt.subplots(figsize=(5.8, 2.9))
    plot = out.rename(columns={"Macro-F1": "Score"})
    model_bar(ax, plot, "Setting", "Score", ["KV-KA", "UV-KA", "KV-UA", "UV-UA"])
    ax.set_ylabel("Macro-F1")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.28))
    savefig(fig, "paper_fig5_ctt_generalization_refined")
    return out


def ablation() -> pd.DataFrame:
    variant_map = {
        "cmf_can": ("Full", "full CMF-CAN"),
        "wo_stats": ("-Win", "without window statistics"),
        "wo_context": ("-Ctx", "without ID context"),
        "wo_xattn": ("-XAttn", "without cross-modal attention"),
        "wo_gate": ("-Gate", "without gated fusion"),
        "concat_fusion": ("Concat", "concatenation fusion"),
        "frame_only": ("Frame", "frame-only"),
        "stats_only": ("Stats", "window statistics only"),
    }
    pd.DataFrame([{"model": k, "short_label": v[0], "description": v[1]} for k, v in variant_map.items()]).to_csv(
        TABLE_DIR / "paper_table_ablation_variant_mapping.csv", index=False
    )
    rows = []
    for dataset, file in [("ROAD", "road_ablation_20ep_merged.csv"), ("CT&T KV-KA", "ctt_ablation_15ep_merged.csv")]:
        df = pd.read_csv(TABLE_DIR / file)
        for model, (short, desc) in variant_map.items():
            hit = df[df["model"].eq(model)]
            rows.append({"Dataset": dataset, "Variant": short, "Description": desc, "F1": np.nan if hit.empty else hit.iloc[0]["f1"], "Macro-F1": np.nan if hit.empty else hit.iloc[0]["macro_f1"]})
    out = pd.DataFrame(rows)
    if (TABLE_DIR / "ctt_unknown_ablation.csv").exists():
        extra = pd.read_csv(TABLE_DIR / "ctt_unknown_ablation.csv")
        for _, r in extra.iterrows():
            short = variant_map.get(r.get("model"), (r.get("model"), ""))[0]
            rows.append({"Dataset": r.get("dataset", "CT&T unknown"), "Variant": short, "Description": variant_map.get(r.get("model"), ("", ""))[1], "F1": r.get("f1"), "Macro-F1": r.get("macro_f1")})
        out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "paper_table_ablation_refined.csv", index=False)
    tex(out, TABLE_DIR / "paper_table_ablation_refined.tex")
    unknown_path = TABLE_DIR / "ctt_unknown_ablation.csv"
    if unknown_path.exists():
        raw = pd.read_csv(unknown_path)
        raw["Variant"] = raw["model"].map(lambda x: variant_map.get(x, (x, ""))[0])
        unk = raw[
            [
                "dataset",
                "model",
                "Variant",
                "f1",
                "macro_f1",
                "aupr",
                "auroc",
                "recall_at_fpr_1em03",
                "f1_at_fpr_1em03",
            ]
        ].rename(
            columns={
                "dataset": "Dataset",
                "model": "Model key",
                "f1": "F1",
                "macro_f1": "Macro-F1",
                "aupr": "AUPR",
                "auroc": "AUROC",
                "recall_at_fpr_1em03": "Recall@FPR=1e-3",
                "f1_at_fpr_1em03": "F1@FPR=1e-3",
            }
        )
        unk.to_csv(TABLE_DIR / "ctt_unknown_ablation_summary.csv", index=False)
        tex(unk, TABLE_DIR / "table_ctt_unknown_ablation.tex")
        datasets_u = ["ctt_test02", "ctt_test03", "ctt_test04"]
        fig, axes = plt.subplots(1, 3, figsize=(10.5, 2.9), sharey=True)
        for ax, ds in zip(axes, datasets_u):
            sub = unk[unk["Dataset"].eq(ds)]
            ax.bar(np.arange(len(sub)), sub["Recall@FPR=1e-3"], color=["#E45756" if v == "Full" else "#D1D5DB" for v in sub["Variant"]], edgecolor="#222222", linewidth=0.8)
            ax.set_title(ds.replace("ctt_", "CT&T "))
            ax.set_xticks(np.arange(len(sub)))
            ax.set_xticklabels(sub["Variant"], rotation=30, ha="right")
            ax.set_ylim(0, 1.0)
            ax.set_ylabel("Recall@FPR=1e-3" if ax is axes[0] else "")
            ax.grid(axis="y")
        savefig(fig, "fig_ctt_unknown_ablation")
    datasets = list(dict.fromkeys(out["Dataset"]))
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.9 * len(datasets), 2.9), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, ds in zip(axes, datasets):
        sub = out[out["Dataset"].eq(ds)]
        bars = ax.bar(np.arange(len(sub)), sub["F1"], color=["#E45756" if v == "Full" else "#D1D5DB" for v in sub["Variant"]], edgecolor="#222222")
        bars[0].set_hatch("")
        bars[0].set_linewidth(1.3)
        ax.set_title(ds)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(np.arange(len(sub)))
        ax.set_xticklabels(sub["Variant"], rotation=30, ha="right")
        ax.set_ylabel("F1" if ax is axes[0] else "")
        ax.grid(axis="y")
    savefig(fig, "paper_fig6_ablation_refined")
    return out


def low_fpr() -> pd.DataFrame:
    base = pd.read_csv(TABLE_DIR / "paper_table_low_fpr_summary.csv")
    recomputed_rows = []
    for pred_path in sorted(PRED_DIR.glob("*_predictions.csv")):
        df = pd.read_csv(pred_path)
        if df.empty or not {"label", "score", "dataset", "model"}.issubset(df.columns):
            continue
        labels = df["label"].to_numpy(int)
        scores = df["score"].to_numpy(float)
        neg = scores[labels == 0]
        if len(neg) == 0:
            continue
        for budget in [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]:
            threshold = float(np.quantile(neg, max(0.0, 1.0 - budget), method="higher"))
            pred = scores >= threshold
            tp = int(((labels == 1) & pred).sum())
            fp = int(((labels == 0) & pred).sum())
            fn = int(((labels == 1) & ~pred).sum())
            tn = int(((labels == 0) & ~pred).sum())
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            recomputed_rows.append(
                {
                    "dataset": df["dataset"].iloc[0],
                    "setting": df["setting"].iloc[0] if "setting" in df.columns else df["dataset"].iloc[0],
                    "model": df["model"].iloc[0],
                    "fpr_budget": budget,
                    "threshold": threshold,
                    "recall": recall,
                    "precision": precision,
                    "f1": 2 * precision * recall / max(precision + recall, 1e-12),
                    "actual_fpr": fp / max(fp + tn, 1),
                    "available_from_scores": True,
                }
            )
    if recomputed_rows:
        pd.DataFrame(recomputed_rows).to_csv(TABLE_DIR / "paper_table_low_fpr_recomputed.csv", index=False)
    recomputed = TABLE_DIR / "paper_table_low_fpr_recomputed.csv"
    if recomputed.exists():
        rec = pd.read_csv(recomputed)
        rec_fmt = rec.rename(
            columns={
                "dataset": "Dataset/Setting",
                "fpr_budget": "fpr_budget_value",
                "recall": "Recall",
            }
        )
        rec_fmt["Model"] = rec_fmt["model"].map(lambda x: MODEL_LABEL.get(x, x))
        rec_fmt["available_budget"] = rec_fmt["fpr_budget_value"].map(lambda x: f"{float(x):g}")
        rec_fmt["source"] = "score_recomputed"
        base_fmt = base.copy()
        base_fmt["source"] = "reported_low_fpr_csv"
        out = pd.concat([base_fmt, rec_fmt[base_fmt.columns.intersection(rec_fmt.columns).tolist()]], ignore_index=True, sort=False)
    else:
        out = base.copy()
    out.fillna("NA").to_csv(TABLE_DIR / "paper_table_low_fpr_refined.csv", index=False)
    tex(out, TABLE_DIR / "paper_table_low_fpr_refined.tex")
    setting_col = "Dataset/Setting" if "Dataset/Setting" in out.columns else "dataset"
    recall_col = "Recall" if "Recall" in out.columns else "recall"
    model_col = "Model" if "Model" in out.columns else "model"
    budget_col = "fpr_budget_value" if "fpr_budget_value" in out.columns else "fpr_budget"
    out_plot = out.copy()
    if model_col == "model":
        out_plot["Model"] = out_plot["model"].map(lambda x: MODEL_LABEL.get(x, x))
        model_col = "Model"
    settings = list(dict.fromkeys(out_plot[setting_col]))
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.3), sharex=True, sharey=True)
    for ax, setting in zip(axes.flat, settings[:4]):
        sub = out_plot[out_plot[setting_col].eq(setting)]
        for model in ["Transformer", "Concat-Fusion", "CMF-CAN"]:
            part = sub[sub[model_col].eq(model)].sort_values(budget_col)
            if part.empty:
                continue
            ax.plot(part[budget_col], part[recall_col], label=model, color=COLORS[model], marker=MARKER[model], linestyle=LINE[model], markeredgecolor="#222222")
        ax.set_title(str(setting).replace("ctt_", ""))
        ax.set_xscale("log")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="both", which="both")
    axes[0, 0].set_ylabel("Recall")
    axes[1, 0].set_ylabel("Recall")
    axes[1, 0].set_xlabel("FPR budget")
    axes[1, 1].set_xlabel("FPR budget")
    axes[1, 1].legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.0, -0.35))
    savefig(fig, "paper_fig7_recall_at_fpr_refined")
    for suffix, selected in [("7a_recall_at_fpr_main", ["ctt_test01", "ctt_test02"]), ("7b_recall_at_fpr_appendix", ["ctt_test03", "ctt_test04"])]:
        sub = out_plot[out_plot[setting_col].isin(selected)]
        if sub.empty:
            continue
        fig, axes = plt.subplots(1, len(selected), figsize=(3.6 * len(selected), 2.65), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, setting in zip(axes, selected):
            g = sub[sub[setting_col].eq(setting)]
            for model in ["Transformer", "Concat-Fusion", "CMF-CAN"]:
                p = g[g[model_col].eq(model)].sort_values(budget_col)
                if not p.empty:
                    ax.plot(p[budget_col], p[recall_col], label=model, color=COLORS[model], marker=MARKER[model], linestyle=LINE[model], markeredgecolor="#222222")
            ax.set_title(setting.replace("ctt_", ""))
            ax.set_xscale("log")
            ax.set_ylim(0, 1.05)
            ax.grid(axis="both", which="both")
        axes[0].set_ylabel("Recall")
        axes[-1].legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.0, -0.28))
        savefig(fig, f"paper_fig{suffix}")
    return out


def efficiency() -> pd.DataFrame:
    eff = pd.read_csv(TABLE_DIR / "efficiency_road.csv")
    main = pd.read_csv(TABLE_DIR / "road_main_20ep.csv")
    models = ["cnn", "lstm", "gru", "transformer", "concat_fusion", "cmf_can"]
    labels = {"cnn": "CNN", "lstm": "LSTM", "gru": "GRU", **MODEL_LABEL}
    rows = []
    for model in models:
        e = eff[eff["model"].eq(model)]
        r = main[main["model"].eq(model)]
        rows.append({"Model": labels[model], "latency_window_us": np.nan if e.empty else e.iloc[0]["avg_batch_ms"] * 1000 / 512, "params": np.nan if e.empty else e.iloc[0]["total_params"], "F1": np.nan if r.empty else r.iloc[0]["f1"], "AUPR": np.nan if r.empty else r.iloc[0]["aupr"]})
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "paper_table_efficiency_refined.csv", index=False)
    tex(out, TABLE_DIR / "paper_table_efficiency_refined.tex")
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for _, r in out.iterrows():
        if pd.isna(r["latency_window_us"]) or pd.isna(r["F1"]):
            continue
        color = COLORS.get(r["Model"], "#9CA3AF")
        ax.scatter(r["latency_window_us"], r["F1"], s=64 if r["Model"] == "CMF-CAN" else 46, color=color, edgecolor="#222222", linewidth=1.2 if r["Model"] == "CMF-CAN" else 0.8)
        ax.text(r["latency_window_us"], r["F1"] + 0.018, r["Model"], ha="center", va="bottom", fontsize=7)
    ax.set_xlabel("Latency per window (us)")
    ax.set_ylabel("ROAD F1")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="both")
    savefig(fig, "paper_fig8_efficiency_tradeoff_refined")
    return out


def dataset_summary() -> pd.DataFrame:
    usage = {
        "road": "main, few-label, ablation, efficiency",
        "ctt_test01": "main, few-label, generalization, ablation, low-FPR",
        "ctt_test02": "unknown vehicle, known attack",
        "ctt_test03": "known vehicle, unknown attack",
        "ctt_test04": "unknown vehicle, unknown attack",
        "crysys_family_mod_subset": "optional external sanity/generalization",
        "hcrl_can_intrusion": "sanity check",
        "car_hacking": "sanity check",
    }
    rows = []
    for ds, use in usage.items():
        root = DATA_DIR / ds
        meta = {}
        for p in root.glob("*_prepare_meta.json") if root.exists() else []:
            meta = json.loads(p.read_text())
            break
        frames = root / "frames.parquet"
        windows = root / "windows_index.npy"
        frame_n = win_n = attack_ratio = vehicles = np.nan
        attacks = "NA"
        if frames.exists():
            fr = pd.read_parquet(frames, columns=["label", "attack_type", "vehicle"])
            frame_n = len(fr)
            attack_ratio = fr["label"].mean()
            vehicles = fr["vehicle"].nunique()
            vals = sorted(str(x) for x in fr["attack_type"].dropna().unique() if str(x) != "normal")
            attacks = ", ".join(vals[:8]) + (" ..." if len(vals) > 8 else "")
        if windows.exists():
            win_n = int(np.load(windows, mmap_mode="r").shape[0])
        rows.append({"Dataset": ds, "Usage": use, "Split protocol": meta.get("split_method") or meta.get("source_test_folder") or "time/group split", "Vehicles": vehicles, "Attack types": attacks, "Train / validation / test setting": "/".join(str(meta.get("split_counts", {}).get(str(i), "NA")) for i in range(3)), "Frames": frame_n, "Windows": win_n, "Attack ratio": attack_ratio, "Notes": meta.get("label_note") or meta.get("frame_label_note") or "NA"})
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "paper_table_dataset_summary_refined.csv", index=False)
    tex(out, TABLE_DIR / "paper_table_dataset_summary_refined.tex")
    return out


def overall_refined() -> pd.DataFrame:
    old = pd.read_csv(TABLE_DIR / "paper_table_overall_main_results.csv")
    old.to_csv(TABLE_DIR / "paper_table_overall_main_results_refined.csv", index=False)
    tex(old, TABLE_DIR / "paper_table_overall_main_results_refined.tex")
    return old


def prediction_tables_and_figs(missing: list[str]) -> None:
    pred_files = sorted(PRED_DIR.glob("*_predictions.csv"))
    gate_files = sorted(PRED_DIR.glob("*_gate_weights.csv"))
    if not pred_files:
        missing.append("Prediction-derived analyses: no prediction CSV files found.")
        return
    preds = pd.concat([pd.read_csv(p) for p in pred_files if p.stat().st_size > 0], ignore_index=True)
    preds["Model"] = preds["model"].map(lambda x: MODEL_LABEL.get(x, x))
    settings_for_curves = ["road", "ctt_test01", "ctt_test02", "ctt_test03", "ctt_test04"]
    if precision_recall_curve and roc_curve:
        for kind, stem, xlab, ylab in [
            ("pr", "paper_fig_pr_curves_road_ctt", "Recall", "Precision"),
            ("roc", "paper_fig_roc_curves_road_ctt", "FPR", "TPR"),
        ]:
            fig, axes = plt.subplots(1, len(settings_for_curves), figsize=(3.0 * len(settings_for_curves), 2.75), sharey=kind == "roc")
            axes = np.atleast_1d(axes)
            for ax, setting in zip(axes, settings_for_curves):
                sub_setting = preds[preds["dataset"].eq(setting)]
                for model in MODEL_ORDER:
                    sub = sub_setting[sub_setting["model"].eq(model)]
                    if sub.empty or sub["label"].nunique() < 2:
                        continue
                    y = sub["label"].to_numpy()
                    s = sub["score"].to_numpy()
                    lab = MODEL_LABEL[model]
                    if kind == "pr":
                        yy, xx, _ = precision_recall_curve(y, s)
                        area = auc(xx, yy)
                    else:
                        xx, yy, _ = roc_curve(y, s)
                        area = auc(xx, yy)
                    ax.plot(xx, yy, label=f"{lab} {area:.3f}", color=COLORS[lab], linestyle=LINE[lab])
                ax.set_title(setting.replace("ctt_", "CT&T "))
                ax.set_xlabel(xlab)
                ax.set_ylabel(ylab if ax is axes[0] else "")
                ax.grid(axis="both")
            axes[-1].legend(frameon=False, fontsize=6.8, loc="upper center", bbox_to_anchor=(0.5, -0.22))
            savefig(fig, stem)

        # Backward-compatible ROAD combined curve.
        road = preds[preds["dataset"].eq("road")]
        if not road.empty:
            fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
            for model in MODEL_ORDER:
                sub = road[road["model"].eq(model)]
                if sub.empty or sub["label"].nunique() < 2:
                    continue
                y = sub["label"].to_numpy()
                s = sub["score"].to_numpy()
                pr, rc, _ = precision_recall_curve(y, s)
                fpr, tpr, _ = roc_curve(y, s)
                lab = MODEL_LABEL[model]
                axes[0].plot(rc, pr, label=f"{lab} ({auc(rc, pr):.3f})", color=COLORS[lab], linestyle=LINE[lab])
                axes[1].plot(fpr, tpr, label=f"{lab} ({auc(fpr, tpr):.3f})", color=COLORS[lab], linestyle=LINE[lab])
            axes[0].set_xlabel("Recall")
            axes[0].set_ylabel("Precision")
            axes[1].set_xlabel("FPR")
            axes[1].set_ylabel("TPR")
            for ax in axes:
                ax.grid(axis="both")
                ax.legend(frameon=False)
            savefig(fig, "paper_fig_pr_roc_curves_road")
    else:
        missing.append("PR/ROC curves: sklearn metrics unavailable.")

    per = []
    fail = []
    for (dataset, model, attack), g in preds.groupby(["dataset", "model", "attack_type"]):
        tp = int(((g["label"] == 1) & (g["prediction"] == 1)).sum())
        fp = int(((g["label"] == 0) & (g["prediction"] == 1)).sum())
        tn = int(((g["label"] == 0) & (g["prediction"] == 0)).sum())
        fn = int(((g["label"] == 1) & (g["prediction"] == 0)).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        fpr = fp / max(fp + tn, 1)
        if attack != "normal":
            per.append({"Dataset": dataset, "Model": MODEL_LABEL[model], "attack_type": attack, "Precision": precision, "Recall": recall, "F1": f1, "FPR": fpr, "count": int(len(g))})
        fail.append({"Dataset": dataset, "Model": MODEL_LABEL[model], "attack_type": attack, "false_negatives": fn, "false_positives": fp, "support": int(len(g))})
    per_df = pd.DataFrame(per)
    if not per_df.empty:
        per_df.to_csv(TABLE_DIR / "paper_table_per_attack_results.csv", index=False)
        tex(per_df, TABLE_DIR / "paper_table_per_attack_results.tex")
        plot = per_df[per_df["Dataset"].isin(["ctt_test02", "ctt_test03", "ctt_test04"])].copy()
        if plot.empty:
            plot = per_df.copy()
        plot["Attack"] = plot["Dataset"].str.replace("ctt_", "T", regex=False) + ":" + plot["attack_type"].astype(str).str.slice(0, 14)
        plot = plot.groupby(["Attack", "Model"], as_index=False).agg(Recall=("Recall", "mean"))
        attacks = list(dict.fromkeys(plot["Attack"]))[:10]
        fig, ax = plt.subplots(figsize=(7.2, 3.0))
        model_bar(ax, plot, "Attack", "Recall", attacks)
        ax.set_ylabel("Recall")
        ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.32))
        savefig(fig, "paper_fig10_per_attack_results")
    fail_df = pd.DataFrame(fail)
    fail_df.to_csv(TABLE_DIR / "paper_table_failure_cases.csv", index=False)
    tex(fail_df, TABLE_DIR / "paper_table_failure_cases.tex")
    top_fail = fail_df.groupby(["Dataset", "Model"], as_index=False)[["false_negatives", "false_positives"]].sum()
    fig, ax = plt.subplots(figsize=(7.0, 2.9))
    top_fail["Setting"] = top_fail["Dataset"].str.replace("ctt_", "CT&T ", regex=False)
    top_fail["total_failures"] = top_fail["false_negatives"] + top_fail["false_positives"]
    model_bar(ax, top_fail, "Setting", "total_failures", list(dict.fromkeys(top_fail["Setting"])))
    ax.set_ylabel("Failure count")
    ax.set_ylim(0, max(1, top_fail["total_failures"].max()) * 1.15)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.28))
    savefig(fig, "paper_fig_failure_cases")

    cal_rows = []
    edges = np.linspace(0.0, 1.0, 11)
    for (dataset, model), g in preds.groupby(["dataset", "model"]):
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (g["score"] >= lo) & (g["score"] < hi if hi < 1.0 else g["score"] <= hi)
            part = g[mask]
            cal_rows.append({"Dataset": dataset, "Model": MODEL_LABEL[model], "bin_low": lo, "bin_high": hi, "count": len(part), "mean_score": np.nan if part.empty else part["score"].mean(), "empirical_attack_rate": np.nan if part.empty else part["label"].mean()})
    cal = pd.DataFrame(cal_rows)
    cal.to_csv(TABLE_DIR / "paper_table_calibration_bins.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 2.8), sharex=True, sharey=True)
    for ax, setting in zip(axes, ["road", "ctt_test02", "ctt_test04"]):
        sub = cal[cal["Dataset"].eq(setting)]
        for model in ["Transformer", "Concat-Fusion", "CMF-CAN"]:
            p = sub[(sub["Model"].eq(model)) & (sub["count"] > 0)]
            if not p.empty:
                ax.plot(p["mean_score"], p["empirical_attack_rate"], marker=MARKER[model], color=COLORS[model], linestyle=LINE[model], label=model)
        ax.plot([0, 1], [0, 1], color="#9CA3AF", linestyle=":", linewidth=1.0)
        ax.set_title(setting.replace("ctt_", "CT&T "))
        ax.set_xlabel("Mean score")
        ax.set_ylabel("Empirical attack rate" if ax is axes[0] else "")
        ax.grid(axis="both")
    axes[-1].legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.0, -0.28))
    savefig(fig, "paper_fig_calibration_reliability")

    if gate_files:
        gates = pd.concat([pd.read_csv(p) for p in gate_files], ignore_index=True)
        table = gates.groupby(["dataset", "setting", "model", "attack_type"], as_index=False)[["gate_frame", "gate_window", "gate_context"]].mean()
        table.to_csv(TABLE_DIR / "paper_table_gate_weights.csv", index=False)
        norm = gates.groupby("dataset", as_index=False)[["gate_frame", "gate_window", "gate_context"]].mean()
        if not norm.empty:
            fig, ax = plt.subplots(figsize=(6.2, 2.9))
            x = np.arange(len(norm))
            width = 0.24
            for i, col in enumerate(["gate_frame", "gate_window", "gate_context"]):
                ax.bar(x + (i - 1) * width, norm[col], width, label=col.replace("gate_", ""), edgecolor="#222222")
            ax.set_xticks(x)
            ax.set_xticklabels(norm["dataset"].str.replace("ctt_", "CT&T ", regex=False), rotation=20, ha="right")
            ax.set_ylabel("Average gate weight")
            ax.set_ylim(0, 1)
            ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.28))
            ax.grid(axis="y")
            savefig(fig, "paper_fig9_gate_weights")
    else:
        missing.append("Gate weight figure: no gate weight CSV exists.")


def write_review(main: pd.DataFrame, ctt: pd.DataFrame, abl: pd.DataFrame, low: pd.DataFrame, eff: pd.DataFrame) -> None:
    road = pd.read_csv(TABLE_DIR / "road_main_20ep.csv")
    t = road[road["model"].eq("transformer")].iloc[0]
    c = road[road["model"].eq("cmf_can")].iloc[0]
    lines = [
        "# CMF-CAN Paper Readiness Review",
        "",
        "## 1. Can the current results support a paper draft?",
        "Yes, but only with a cautious mixed-results narrative. The current package is enough for a first draft around cross-modality CAN IDS and deployment-oriented analysis, not enough for a universal superiority claim.",
        "",
        "## 2. Does it support 'CMF-CAN comprehensively outperforms Transformer'?",
        f"No. On ROAD, Transformer has higher F1/Macro-F1 ({t['f1']:.4f}/{t['macro_f1']:.4f}) than CMF-CAN ({c['f1']:.4f}/{c['macro_f1']:.4f}), while CMF-CAN improves AUROC/AUPR ({c['auroc']:.4f}/{c['aupr']:.4f} vs {t['auroc']:.4f}/{t['aupr']:.4f}).",
        "",
        "## 3. Best narrative",
        "Recommended: cross-modality feature fusion for CAN IDS with deployment-oriented low-FPR evidence. Label efficiency is a secondary, setting-dependent result. Generalizable CAN IDS should not be the main claim.",
        "",
        "## 4. Strongest results",
        "- CT&T KV-KA: CMF-CAN slightly improves F1/Macro-F1 over Transformer.",
        "- CT&T UV-KA low-FPR: CMF-CAN shows a large recall advantage at measured FPR budgets.",
        "- CT&T unknown-setting ablation: window statistics and some simplified variants transfer better than the full model in selected shifted settings.",
        "- ROAD ranking metrics: CMF-CAN improves AUROC/AUPR over Transformer despite weaker thresholded F1.",
        "- CrySyS-subset: all three main models are close; CMF-CAN is competitive, not decisively dominant.",
        "",
        "## 5. Weakest results",
        "- CT&T KV-UA and UV-UA have low absolute F1, especially test04.",
        "- ROAD F1/Macro-F1 favor Transformer.",
        "- Ablation does not show Full CMF-CAN is always the best thresholded-F1 or low-FPR variant; test02/test04 often favor simplified variants.",
        "- HCRL and Car-Hacking are near-saturated sanity checks and should not be overemphasized.",
        "",
        "## 6. Main-paper figures",
        "Figures 1-8 refined are suitable for the main paper if the text states dataset-dependent behavior. Figure 7a is preferable for the main low-FPR story; Figure 7b belongs in the appendix/discussion.",
        "",
        "## 7. Appendix figures",
        "Shifted PR/ROC, gate weights, failure cases, calibration bins, per-attack summaries and t-SNE embeddings can go to appendix. HCRL/Car-Hacking should be appendix sanity checks.",
        "",
        "## 8. Conclusions that cannot be written",
        "- Do not claim consistent superiority over Transformer across all datasets and metrics.",
        "- Do not claim unknown vehicle + unknown attack generalization is solved.",
        "- Do not claim every modality is always beneficial.",
        "- Do not claim CT&T few-label superiority is stable at every label ratio.",
        "- Do not claim gated fusion or ID context is always beneficial under unknown vehicle/attack shifts.",
        "",
        "## 9. Remaining key experiments",
        "- P0 evidence is now complete for a draft package: CT&T shifted prediction dumps, gate dumps, unknown-setting ablation, shifted PR/ROC, failure cases, calibration bins and sampled embeddings.",
        "- Remaining high-value additions are multi-seed unknown-setting ablations and full reproduction of recent external baselines.",
        "",
        "## 10. CCF B/C or intelligent vehicle security readiness",
        "Basically enough for an initial submission draft if framed as a systematic cross-modality/deployment study and if limitations are explicit.",
        "",
        "## 11. CCF A / top-tier security gap",
        "Still short. Needs stronger unknown-shift results, complete per-sample evidence, more rigorous baselines, more seeds for shifted CT&T, deployment validation, and stronger ablation under unknown vehicle/attack shifts.",
    ]
    (TABLE_DIR / "paper_readiness_review.md").write_text("\n".join(lines) + "\n")


def writing_guidance() -> None:
    lines = [
        "# Paper Writing Guidance",
        "",
        "## Can Write",
        "- CMF-CAN provides a structured way to integrate frame-level sequence, window-level statistics and ID-context features.",
        "- On ROAD, CMF-CAN improves ranking-oriented AUROC/AUPR over Transformer, although thresholded F1 is lower.",
        "- On CT&T known vehicle + known attack, CMF-CAN is competitive and slightly stronger in F1/Macro-F1.",
        "- In CT&T unknown vehicle + known attack low-FPR analysis, CMF-CAN has a clear recall advantage over Transformer and Concat-Fusion.",
        "- CT&T unknown-setting ablation shows window-level statistics are a robust shifted-setting signal; removing window statistics usually hurts low-FPR behavior.",
        "- Unknown-setting ablation also shows Full CMF-CAN is not always the best deployable variant, so the contribution should be framed as a system study of cross-modality fusion rather than a universal model win.",
        "",
        "## Cannot Write",
        "- Do not claim CMF-CAN consistently outperforms Transformer.",
        "- Do not claim CMF-CAN solves unknown attack or unknown vehicle + unknown attack generalization.",
        "- Do not claim Full CMF-CAN is always the best ablation variant.",
        "- Do not claim ID context is always beneficial; unknown vehicle settings can favor variants without ID context.",
        "- Do not present HCRL/Car-Hacking as hard evidence; they are sanity checks.",
        "",
        "## Recommended Positioning",
        "Best fit: Cross-modality feature fusion for CAN IDS, with deployment-oriented low-FPR operating analysis. Label-efficient CAN IDS can be a supporting angle, not the headline.",
        "",
        "## Strict Claim Guardrails",
        "- Prefer: 'CMF-CAN and its ablations reveal which modalities transfer under each CT&T shift.'",
        "- Prefer: 'Window statistics are the most stable shifted-setting signal in our CT&T unknown ablation.'",
        "- Avoid: 'CMF-CAN consistently solves generalization.'",
        "- Avoid: 'Cross-modal attention/gating is always beneficial.'",
        "",
        "## Risks",
        "- ROAD F1/Macro-F1 are worse than Transformer.",
        "- Few-label results are mixed across ROAD and CT&T.",
        "- Full CMF-CAN is not always best in ablation.",
        "- CT&T test03/test04 have low absolute F1.",
        "- HCRL/Car-Hacking are too easy and should be appendix sanity checks.",
        "",
        "## Recommended Additional Experiments",
        "P0: complete for this package: CT&T prediction/gate dumps, unknown-setting ablation, shifted PR/ROC inputs, and low-FPR recomputation from scores.",
        "P1: remaining optional work: more seeds for unknown ablation and external industrial baselines.",
        "P2: longer-term work: more external datasets and full reproduction of recent sequence/state-space baselines.",
    ]
    (TABLE_DIR / "paper_writing_guidance.md").write_text("\n".join(lines) + "\n")


def missing_report(missing: list[str]) -> None:
    expected_preds = [f"{ds}_{m}_predictions.csv" for ds in ["road", "ctt_test01", "ctt_test02", "ctt_test03", "ctt_test04"] for m in MODEL_ORDER]
    expected_gates = [f"{ds}_cmf_can_gate_weights.csv" for ds in ["road", "ctt_test01", "ctt_test02", "ctt_test03", "ctt_test04"]]
    missing += [f"Missing prediction dump: {x}" for x in expected_preds if not (PRED_DIR / x).exists()]
    missing += [f"Missing gate weight dump: {x}" for x in expected_gates if not (PRED_DIR / x).exists()]
    if not (Path("results/cmf_embeddings") / "road_cmf_can_embedding_sample.npy").exists():
        missing.append("Missing embedding dumps for UMAP/t-SNE.")
    if not (TABLE_DIR / "ctt_unknown_ablation.csv").exists():
        missing.append("Missing CT&T unknown-setting ablation CSV for test02/test03/test04.")
    if not (TABLE_DIR / "paper_table_calibration_bins.csv").exists():
        missing.append("Missing bin-level calibration table.")
    missing.append("Remaining methodological limitation: CT&T unknown-setting ablation is single-seed eval-only from CT&T test01 checkpoints; add multi-seed retraining for stronger top-tier claims.")
    lines = ["# Missing Inputs Report", "", "No fake figures were generated.", ""]
    unique = list(dict.fromkeys(missing))
    if unique:
        lines += ["## Missing or Partial Inputs", "", *[f"- {m}" for m in unique]]
    else:
        lines += ["## Missing or Partial Inputs", "", "- None for requested P0/P1 evidence files in the current package."]
    lines += [
        "",
        "## Completed Evidence Files",
        "- CT&T test02-test04 prediction dumps for Transformer, Concat-Fusion and CMF-CAN exist.",
        "- CT&T test02-test04 CMF-CAN gate weight dumps exist.",
        "- CT&T unknown-setting ablation table exists.",
        "- Shifted PR/ROC, failure-case, per-attack, calibration-bin and gate-weight figures are generated from real dumps.",
        "",
        "## Reproduction Commands",
        "- Prediction/gate dump: `python -m cmf_can.training.cli --dataset <dataset> --model <model> --eval-only --save-predictions [--save-gate-weights] --num-workers 0`.",
        "- Embedding dump: `python -m cmf_can.analysis.export_embeddings --datasets road ctt_test02 ctt_test04 --model cmf_can`.",
        "- Figure/table refresh: `python -m cmf_can.analysis.export_paper_refined_assets --root .`.",
        "",
        "## Recommended Saved Fields",
        "sample_id, dataset, setting, model, label, prediction, score, attack_type, vehicle, window_start, window_end, split, gate_frame, gate_window, gate_context, embedding_path.",
    ]
    (TABLE_DIR / "missing_inputs_report.md").write_text("\n".join(lines) + "\n")


def inventory() -> None:
    figs = [
        ("Figure 1", "CMF-CAN architecture", "manual refined architecture", "paper_fig1_architecture_refined.*", "main paper", "Core cross-modality pipeline", "Post-processing branches intentionally omitted"),
        ("Figure 2", "Main multi-dataset results", "road_main_20ep.csv; ctt_generalization_15ep.csv; CrySyS optional", "paper_fig2_main_multidataset_refined.*", "main paper", "Mixed main performance", "Macro-F1 fallback to F1 where needed"),
        ("Figure 3", "ROAD few-label", "road_few_label_3seed_mean_std.csv", "paper_fig3_road_few_label_refined.*", "main paper", "Label efficiency is mixed", "Transformer wins several ROAD ratios"),
        ("Figure 4", "CT&T few-label", "ctt_few_label_3seed_mean_std.csv", "paper_fig4_ctt_few_label_refined.*", "main paper", "Few-label is setting-dependent", "Not stable dominance"),
        ("Figure 5", "CT&T generalization", "ctt_generalization_15ep.csv", "paper_fig5_ctt_generalization_refined.*", "main paper", "Unknown settings are hard", "test03/test04 low absolute F1"),
        ("Figure 6", "Ablation", "road_ablation_20ep_merged.csv; ctt_ablation_15ep_merged.csv", "paper_fig6_ablation_refined.*", "main paper", "Fusion components help in selected settings", "Full not always best"),
        ("Figure 7", "Recall@FPR", "paper_table_low_fpr_summary.csv", "paper_fig7*_recall_at_fpr*.{png,pdf,svg}", "main/appendix", "UV-KA low-FPR is strongest", "Only measured budgets unless recomputed"),
        ("Figure 8", "Efficiency trade-off", "efficiency_road.csv; road_main_20ep.csv", "paper_fig8_efficiency_tradeoff_refined.*", "main/appendix", "CMF-CAN has acceptable overhead", "Slightly slower than Transformer"),
        ("Figure 9", "Gate weights", "results/cmf_predictions/*gate_weights.csv", "paper_fig9_gate_weights.*", "appendix", "Gate interpretability from completed dumps", "ROAD only until CT&T gates exist"),
        ("Figure 10", "Per-attack results", "results/cmf_predictions/*predictions.csv", "paper_fig10_per_attack_results.*", "appendix", "Attack-level evidence from completed dumps", "Attack labels follow processed dataset labels"),
        ("Appendix", "PR curves", "results/cmf_predictions/*predictions.csv", "paper_fig_pr_curves_road_ctt.*", "appendix", "Ranking behavior across shifted settings", "Can be optimistic under threshold shift"),
        ("Appendix", "ROC curves", "results/cmf_predictions/*predictions.csv", "paper_fig_roc_curves_road_ctt.*", "appendix", "Ranking behavior across shifted settings", "Use with low-FPR curves for deployment"),
        ("Appendix", "Failure cases", "results/cmf_predictions/*predictions.csv", "paper_fig_failure_cases.*", "appendix", "False positive/negative burden by setting", "Counts depend on setting size"),
        ("Appendix", "Calibration reliability", "results/cmf_predictions/*predictions.csv", "paper_fig_calibration_reliability.*", "appendix", "Score calibration bins", "Post-hoc calibration not applied here"),
        ("Appendix", "t-SNE embeddings", "results/cmf_embeddings/*embedding_sample.*", "paper_fig_tsne_embeddings.*", "appendix", "Representation separability sample", "Sampled visualization only"),
    ]
    tables = [
        "paper_table_dataset_summary_refined",
        "paper_table_overall_main_results_refined",
        "paper_table_few_label_refined",
        "paper_table_ctt_generalization_refined",
        "paper_table_low_fpr_refined",
        "paper_table_ablation_refined",
        "paper_table_efficiency_refined",
    ]
    lines = ["# Paper Figure/Table Inventory", "", "| Figure ID | Figure name | Input files | Output files | Recommended placement | Main message | Caveats |", "|---|---|---|---|---|---|---|"]
    for row in figs:
        lines.append("| " + " | ".join(row) + " |")
    lines += ["", "| Table | Output files | Recommended placement | Caveats |", "|---|---|---|---|"]
    for name in tables:
        lines.append(f"| {name} | `{name}.csv`, `{name}.tex` | main/appendix as appropriate | NA retained; no imputation |")
    (TABLE_DIR / "paper_figure_table_inventory.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    os.chdir(args.root.resolve())
    setup_style()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    missing: list[str] = []
    fig_architecture()
    main_df = main_results()
    road_few = few_label("road", "road_few_label_3seed_mean_std.csv", "paper_fig3_road_few_label_refined", "paper_table_few_label_refined")
    ctt_few = few_label("ctt_test01", "ctt_few_label_3seed_mean_std.csv", "paper_fig4_ctt_few_label_refined", "paper_table_ctt_few_label_summary_refined")
    few_all = pd.concat([road_few.assign(Dataset="ROAD"), ctt_few.assign(Dataset="CT&T KV-KA")], ignore_index=True)
    few_all.to_csv(TABLE_DIR / "paper_table_few_label_refined.csv", index=False)
    tex(few_all, TABLE_DIR / "paper_table_few_label_refined.tex")
    ctt_df = ctt_generalization()
    abl_df = ablation()
    low_df = low_fpr()
    eff_df = efficiency()
    dataset_summary()
    overall_refined()
    prediction_tables_and_figs(missing)
    write_review(main_df, ctt_df, abl_df, low_df, eff_df)
    writing_guidance()
    missing_report(missing)
    inventory()
    print("[write] refined CMF-CAN paper evidence package")


if __name__ == "__main__":
    main()
