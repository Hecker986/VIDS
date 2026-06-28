from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


TABLE_DIR = Path("results/cmf_tables")
FIG_DIR = Path("results/cmf_figures")
DATA_DIR = Path("data/processed")

MODEL_ORDER = ["transformer", "concat_fusion", "cmf_can"]
MODEL_LABELS = {
    "cnn": "CNN",
    "lstm": "LSTM",
    "gru": "GRU",
    "transformer": "Transformer",
    "concat_fusion": "Concat-Fusion",
    "cmf_can": "CMF-CAN",
}
MODEL_INV = {v: k for k, v in MODEL_LABELS.items()}

TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}
MODEL_STYLE = {
    "Transformer": {"color": "#A3BEFA", "edge": "#2E4780", "hatch": "///", "marker": "o", "ls": "-"},
    "Concat-Fusion": {"color": "#FFE15B", "edge": "#736422", "hatch": "\\\\\\", "marker": "s", "ls": "--"},
    "CMF-CAN": {"color": "#F0986E", "edge": "#804126", "hatch": "", "marker": "^", "ls": ":"},
}


def set_style() -> None:
    sns.set_theme(
        style="whitegrid",
        context="paper",
        rc={
            "figure.facecolor": TOKENS["surface"],
            "axes.facecolor": TOKENS["panel"],
            "axes.edgecolor": TOKENS["axis"],
            "axes.labelcolor": TOKENS["ink"],
            "xtick.color": TOKENS["ink"],
            "ytick.color": TOKENS["ink"],
            "grid.color": TOKENS["grid"],
            "grid.linewidth": 0.8,
        },
    )
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "legend.fontsize": 7.8,
            "figure.titlesize": 11.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.9,
            "lines.linewidth": 1.5,
            "patch.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def savefig(fig: mpl.figure.Figure, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.035}
        if ext == "png":
            kwargs["dpi"] = 300
        fig.savefig(FIG_DIR / f"{stem}.{ext}", **kwargs)
    plt.close(fig)


def add_header(fig: mpl.figure.Figure, title: str, subtitle: str, *, y: float = 0.98) -> None:
    fig.text(0.06, y, title, ha="left", va="top", fontsize=11.5, fontweight="bold", color=TOKENS["ink"])
    fig.text(0.06, y - 0.075, subtitle, ha="left", va="top", fontsize=8.3, color=TOKENS["muted"])


def fmt_num(x: object, digits: int = 4) -> str:
    if x is None or pd.isna(x):
        return "NA"
    return f"{float(x):.{digits}f}"


def fmt_mean_std(mean: object, std: object, digits: int = 3) -> str:
    if mean is None or pd.isna(mean):
        return "NA"
    if std is None or pd.isna(std):
        return f"{float(mean):.{digits}f}"
    return f"{float(mean):.{digits}f} $\\pm$ {float(std):.{digits}f}"


def write_tex(df: pd.DataFrame, path: Path) -> None:
    df.to_latex(path, index=False, escape=False, na_rep="NA")


def copy_architecture(missing: list[str]) -> None:
    for ext in ("png", "pdf", "svg"):
        src = FIG_DIR / f"fig_model_architecture_cmf_can.{ext}"
        dst = FIG_DIR / f"paper_fig1_architecture.{ext}"
        if src.exists():
            shutil.copyfile(src, dst)
        else:
            missing.append(f"paper_fig1_architecture.{ext}: missing source {src}")


def read_rows(path: Path, dataset_label: str, setting_filter: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if setting_filter and "dataset" in df:
        df = df[df["dataset"] == setting_filter]
    df = df[df["model"].isin(MODEL_ORDER)].copy()
    df["Dataset/Setting"] = dataset_label
    df["Model"] = df["model"].map(MODEL_LABELS)
    return df


def collect_main_results() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    specs = [
        (TABLE_DIR / "road_main_20ep.csv", "ROAD", None),
        (TABLE_DIR / "ctt_generalization_15ep.csv", "CT&T test01", "ctt_test01"),
        (TABLE_DIR / "hcrl_main_15ep.csv", "HCRL", None),
        (TABLE_DIR / "car_hacking_main_15ep.csv", "Car-Hacking", None),
    ]
    for path, label, filt in specs:
        if path.exists():
            frames.append(read_rows(path, label, filt))

    cry_path = TABLE_DIR / "crysys_family_mod_3model_3seed_mean_std.csv"
    if cry_path.exists():
        cry = pd.read_csv(cry_path)
        piv = cry[cry["model"].isin(MODEL_ORDER)].pivot_table(index="model", columns="metric", values=["mean", "std"], aggfunc="first")
        rows = []
        for model in MODEL_ORDER:
            rows.append(
                {
                    "Dataset/Setting": "CrySyS",
                    "model": model,
                    "Model": MODEL_LABELS[model],
                    "precision": piv.loc[model, ("mean", "precision")],
                    "recall": piv.loc[model, ("mean", "recall")],
                    "f1": piv.loc[model, ("mean", "f1")],
                    "macro_f1": piv.loc[model, ("mean", "macro_f1")],
                    "auroc": piv.loc[model, ("mean", "auroc")],
                    "aupr": piv.loc[model, ("mean", "aupr")],
                    "fpr": piv.loc[model, ("mean", "fpr")],
                    "fnr": piv.loc[model, ("mean", "fnr")],
                    "f1_std": piv.loc[model, ("std", "f1")],
                    "macro_f1_std": piv.loc[model, ("std", "macro_f1")],
                }
            )
        frames.append(pd.DataFrame(rows))

    out = pd.concat(frames, ignore_index=True, sort=False)
    cols = ["Dataset/Setting", "Model", "precision", "recall", "f1", "macro_f1", "auroc", "aupr", "fpr", "fnr"]
    out[cols].to_csv(TABLE_DIR / "paper_table_overall_main_results.csv", index=False)
    tex = out[cols].copy()
    for c in cols[2:]:
        tex[c] = tex[c].map(lambda x: fmt_num(x, 4))
    write_tex(tex.rename(columns={c: c.upper() if c in ["f1", "fpr", "fnr"] else c.replace("_", "-").upper() for c in cols[2:]}), TABLE_DIR / "paper_table_overall_main_results.tex")
    return out


def fig_main_multidataset(main: pd.DataFrame) -> None:
    plot = main[["Dataset/Setting", "Model", "f1", "macro_f1", "f1_std", "macro_f1_std"]].copy()
    plot.to_csv(TABLE_DIR / "paper_table_main_multidataset.csv", index=False)
    datasets = list(dict.fromkeys(plot["Dataset/Setting"]))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
    for ax, metric, title in zip(axes, ["f1", "macro_f1"], ["F1", "Macro-F1"]):
        grouped_bar(ax, plot, datasets, "Model", metric, f"{metric}_std")
        ax.set_title(title)
        ax.set_ylabel("Score" if ax is axes[0] else "")
    add_header(fig, "Main multi-dataset performance", "Grouped bars compare Transformer, Concat-Fusion and CMF-CAN; NA rows are retained in CSV tables.")
    legend_for_models(axes[-1], y=-0.27)
    fig.subplots_adjust(top=0.72, bottom=0.28, wspace=0.12)
    savefig(fig, "paper_fig2_main_multidataset")


def grouped_bar(ax: mpl.axes.Axes, df: pd.DataFrame, categories: list[str], series_col: str, value_col: str, std_col: str | None = None) -> None:
    series = ["Transformer", "Concat-Fusion", "CMF-CAN"]
    x = np.arange(len(categories))
    width = 0.23
    for i, name in enumerate(series):
        sub = df[df[series_col] == name].set_index("Dataset/Setting").reindex(categories)
        style = MODEL_STYLE[name]
        values = sub[value_col].astype(float)
        ax.bar(
            x + (i - 1) * width,
            values,
            width,
            color=style["color"],
            edgecolor=style["edge"],
            hatch=style["hatch"],
            linewidth=1.1 if name == "CMF-CAN" else 0.8,
            label=name,
        )
        if std_col and std_col in sub:
            ax.errorbar(x + (i - 1) * width, values, yerr=sub[std_col].fillna(0), fmt="none", ecolor="#334155", capsize=2, elinewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=18, ha="right")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="x", visible=False)


def legend_for_models(ax: mpl.axes.Axes, y: float = -0.18) -> None:
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:3], labels[:3], frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.0, y), columnspacing=1.5)


def few_label_table(path: Path, fallback_paths: list[Path], dataset: str, out_name: str) -> pd.DataFrame:
    if path.exists():
        df = pd.read_csv(path)
        if "dataset" in df:
            df = df[df["dataset"].eq(dataset)]
        out = df[df["model"].isin(MODEL_ORDER)].copy()
        out["Model"] = out["model"].map(MODEL_LABELS)
        out = out[["label_ratio", "Model", "f1_mean", "f1_std", "precision_mean", "recall_mean", "auroc_mean", "aupr_mean"]]
    else:
        parts = []
        for fp in fallback_paths:
            if fp.exists():
                d = pd.read_csv(fp)
                if "dataset" in d:
                    d = d[d["dataset"].eq(dataset)]
                parts.append(d[d["model"].isin(MODEL_ORDER)])
        raw = pd.concat(parts, ignore_index=True)
        grp = raw.groupby(["label_ratio", "model"], as_index=False).agg(f1_mean=("f1", "mean"), f1_std=("f1", "std"), precision_mean=("precision", "mean"), recall_mean=("recall", "mean"), auroc_mean=("auroc", "mean"), aupr_mean=("aupr", "mean"))
        grp["Model"] = grp["model"].map(MODEL_LABELS)
        out = grp[["label_ratio", "Model", "f1_mean", "f1_std", "precision_mean", "recall_mean", "auroc_mean", "aupr_mean"]]
    out = out.sort_values(["label_ratio", "Model"])
    out.to_csv(TABLE_DIR / out_name, index=False)
    return out


def fig_few_label(df: pd.DataFrame, stem: str, title: str, table_tex: str | None = None) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 2.75))
    for model in ["Transformer", "Concat-Fusion", "CMF-CAN"]:
        sub = df[df["Model"] == model].sort_values("label_ratio")
        if sub.empty:
            continue
        style = MODEL_STYLE[model]
        x = sub["label_ratio"].to_numpy(dtype=float) * 100
        y = sub["f1_mean"].to_numpy(dtype=float)
        ax.plot(x, y, marker=style["marker"], linestyle=style["ls"], color=style["color"], markeredgecolor=style["edge"], label=model)
        if "f1_std" in sub and sub["f1_std"].notna().any():
            std = sub["f1_std"].fillna(0).to_numpy(dtype=float)
            ax.fill_between(x, np.clip(y - std, 0, 1), np.clip(y + std, 0, 1), color=style["color"], alpha=0.16, linewidth=0)
    ax.set_xscale("log")
    ax.set_xticks([1, 5, 10, 20, 100])
    ax.set_xticklabels(["1%", "5%", "10%", "20%", "100%"])
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Labeled training ratio")
    ax.set_ylabel("F1")
    ax.grid(axis="x", visible=True, which="both", linewidth=0.5)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    add_header(fig, title, "Mean F1 with shaded ±1 std band when multi-seed summaries are available.", y=1.02)
    fig.subplots_adjust(top=0.74, bottom=0.3)
    savefig(fig, stem)
    if table_tex:
        tex = df.copy()
        tex["F1"] = [fmt_mean_std(m, s) for m, s in zip(tex["f1_mean"], tex["f1_std"])]
        write_tex(tex[["label_ratio", "Model", "F1"]].rename(columns={"label_ratio": "Label ratio"}), TABLE_DIR / table_tex)


def fig_and_table_few_label() -> None:
    road = few_label_table(
        TABLE_DIR / "road_few_label_3seed_mean_std.csv",
        [TABLE_DIR / "road_few_label_seed42_20ep.csv", TABLE_DIR / "road_few_label_seed2024_20ep.csv", TABLE_DIR / "road_few_label_seed2026_20ep.csv"],
        "road",
        "paper_table_road_few_label.csv",
    )
    fig_few_label(road, "paper_fig3_road_few_label", "ROAD few-label performance")
    ctt = few_label_table(TABLE_DIR / "ctt_few_label_3seed_mean_std.csv", [TABLE_DIR / "ctt_few_label_15ep.csv"], "ctt_test01", "paper_table_ctt_few_label.csv")
    fig_few_label(ctt, "paper_fig4_ctt_few_label", "CT&T test01 few-label performance")
    few = pd.concat([road.assign(Dataset="ROAD"), ctt.assign(Dataset="CT&T test01")], ignore_index=True)
    few.to_csv(TABLE_DIR / "paper_table_few_label_summary.csv", index=False)
    tex = few.copy()
    tex["F1"] = [fmt_mean_std(m, s) for m, s in zip(tex["f1_mean"], tex["f1_std"])]
    write_tex(tex[["Dataset", "label_ratio", "Model", "F1"]].rename(columns={"label_ratio": "Label ratio"}), TABLE_DIR / "paper_table_few_label_summary.tex")


def fig_ctt_generalization() -> pd.DataFrame:
    mapping = {
        "ctt_test01": ("T01 KV+KA", "known vehicle + known attack"),
        "ctt_test02": ("T02 UV+KA", "unknown vehicle + known attack"),
        "ctt_test03": ("T03 KV+UA", "known vehicle + unknown attack"),
        "ctt_test04": ("T04 UV+UA", "unknown vehicle + unknown attack"),
    }
    df = pd.read_csv(TABLE_DIR / "ctt_generalization_15ep.csv")
    df = df[df["model"].isin(MODEL_ORDER)].copy()
    df["Dataset/Setting"] = df["dataset"].map(lambda x: mapping[x][0])
    df["Setting description"] = df["dataset"].map(lambda x: mapping[x][1])
    df["Model"] = df["model"].map(MODEL_LABELS)
    rows = []
    for setting, part in df.groupby("Dataset/Setting", sort=False):
        base = part.loc[part["model"] == "transformer"]
        base_f1 = base["f1"].iloc[0] if not base.empty else np.nan
        base_macro = base["macro_f1"].iloc[0] if not base.empty else np.nan
        for rec in part.itertuples():
            rows.append(
                {
                    "Dataset/Setting": setting,
                    "Setting description": part["Setting description"].iloc[0],
                    "Model": MODEL_LABELS[rec.model],
                    "F1": rec.f1,
                    "Macro-F1": rec.macro_f1,
                    "F1 improvement vs Transformer": np.nan if pd.isna(base_f1) or base_f1 == 0 else (rec.f1 - base_f1) / base_f1,
                    "Macro-F1 improvement vs Transformer": np.nan if pd.isna(base_macro) or base_macro == 0 else (rec.macro_f1 - base_macro) / base_macro,
                    "AUROC": rec.auroc,
                    "AUPR": rec.aupr,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "paper_table_ctt_generalization.csv", index=False)
    out.to_csv(TABLE_DIR / "paper_table_ctt_generalization_summary.csv", index=False)
    tex = out.copy()
    for c in ["F1", "Macro-F1", "F1 improvement vs Transformer", "Macro-F1 improvement vs Transformer", "AUROC", "AUPR"]:
        tex[c] = tex[c].map(lambda x: fmt_num(x, 4))
    write_tex(tex, TABLE_DIR / "paper_table_ctt_generalization_summary.tex")

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0), sharey=True)
    plot = out.rename(columns={"F1": "f1", "Macro-F1": "macro_f1"})
    cats = [v[0] for v in mapping.values()]
    for ax, metric, title in zip(axes, ["f1", "macro_f1"], ["F1", "Macro-F1"]):
        grouped_bar(ax, plot, cats, "Model", metric, None)
        ax.set_title(title)
        ax.set_ylabel("Score" if ax is axes[0] else "")
    add_header(fig, "CT&T known/unknown generalization", "KV/UV = known/unknown vehicle; KA/UA = known/unknown attack. Table includes relative improvement vs Transformer.")
    legend_for_models(axes[-1], y=-0.28)
    fig.subplots_adjust(top=0.72, bottom=0.34, wspace=0.12)
    savefig(fig, "paper_fig5_ctt_generalization")
    return out


def fig_ablation(missing: list[str]) -> None:
    order = [
        ("cmf_can", "Full CMF-CAN"),
        ("wo_stats", "w/o window stats"),
        ("wo_context", "w/o ID context"),
        ("wo_xattn", "w/o cross-modal attn."),
        ("wo_gate", "w/o gated fusion"),
        ("concat_fusion", "Concat fusion"),
        ("frame_only", "Frame-only"),
        ("stats_only", "Stats-only"),
    ]
    files = [("ROAD", TABLE_DIR / "road_ablation_20ep_merged.csv", TABLE_DIR / "road_ablation_20ep.csv"), ("CT&T test01", TABLE_DIR / "ctt_ablation_15ep_merged.csv", TABLE_DIR / "ctt_ablation_15ep.csv")]
    rows = []
    for dataset, preferred, fallback in files:
        path = preferred if preferred.exists() else fallback
        df = pd.read_csv(path)
        present = set(df["model"])
        for model, label in order:
            if model not in present:
                missing.append(f"paper_fig6_ablation: missing variant {label} for {dataset}")
                rows.append({"Dataset": dataset, "Variant": label, "F1": np.nan, "Macro-F1": np.nan})
                continue
            rec = df[df["model"] == model].iloc[0]
            rows.append({"Dataset": dataset, "Variant": label, "F1": rec["f1"], "Macro-F1": rec["macro_f1"]})
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "paper_table_ablation.csv", index=False)
    tex = out.copy()
    for c in ["F1", "Macro-F1"]:
        tex[c] = tex[c].map(lambda x: fmt_num(x, 4))
    write_tex(tex, TABLE_DIR / "paper_table_ablation.tex")

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0), sharey=True)
    colors = ["#F0986E", "#A3BEFA", "#A3BEFA", "#A3BEFA", "#A3BEFA", "#FFE15B", "#C5CAD3", "#E2E5EA"]
    hatches = ["", "///", "///", "///", "///", "\\\\\\", "", ""]
    for ax, dataset in zip(axes, ["ROAD", "CT&T test01"]):
        sub = out[out["Dataset"] == dataset]
        bars = ax.bar(np.arange(len(sub)), sub["F1"], color=colors, edgecolor="#464C55", linewidth=0.8)
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)
        bars[0].set_linewidth(1.3)
        bars[0].set_edgecolor("#804126")
        ax.set_xticks(np.arange(len(sub)))
        ax.set_xticklabels(sub["Variant"], rotation=35, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_title(dataset)
        ax.set_ylabel("F1" if ax is axes[0] else "")
        ax.grid(axis="x", visible=False)
    add_header(fig, "Modality and fusion ablation", "Grouped hatch-style bars compare available CMF-CAN variants; missing variants are reported separately.")
    fig.subplots_adjust(top=0.72, bottom=0.37, wspace=0.12)
    savefig(fig, "paper_fig6_ablation")


def fig_recall_at_fpr(missing: list[str]) -> pd.DataFrame:
    budgets = [("1e-4", "recall_at_fpr_1em04"), ("5e-4", "recall_at_fpr_5em04"), ("1e-3", "recall_at_fpr_1em03")]
    unavailable = ["5e-3", "1e-2"]
    for budget in unavailable:
        missing.append(f"paper_fig7_recall_at_fpr: FPR budget {budget} unavailable in current CSVs; prediction scores are required.")
    df = pd.read_csv(TABLE_DIR / "ctt_deployment_low_fpr_15ep.csv")
    df = df[df["model"].isin(MODEL_ORDER)].copy()
    rows = []
    for rec in df.itertuples():
        for budget, col in budgets:
            rows.append(
                {
                    "Dataset/Setting": rec.dataset,
                    "Model": MODEL_LABELS[rec.model],
                    "available_budget": budget,
                    "fpr_budget_value": float(budget),
                    "Recall": getattr(rec, col),
                    "AUPR": rec.aupr,
                    "AUROC": rec.auroc,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "paper_table_recall_at_fpr.csv", index=False)
    out.to_csv(TABLE_DIR / "paper_table_low_fpr_summary.csv", index=False)
    tex = out.copy()
    for c in ["Recall", "AUPR", "AUROC"]:
        tex[c] = tex[c].map(lambda x: fmt_num(x, 4))
    write_tex(tex, TABLE_DIR / "paper_table_low_fpr_summary.tex")

    settings = ["ctt_test01", "ctt_test02", "ctt_test03", "ctt_test04"]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.4), sharex=True, sharey=True)
    for ax, setting in zip(axes.flat, settings):
        sub = out[out["Dataset/Setting"] == setting]
        for model in ["Transformer", "Concat-Fusion", "CMF-CAN"]:
            part = sub[sub["Model"] == model].sort_values("fpr_budget_value")
            if part.empty:
                continue
            st = MODEL_STYLE[model]
            ax.plot(part["fpr_budget_value"], part["Recall"], label=model, color=st["color"], marker=st["marker"], linestyle=st["ls"], markeredgecolor=st["edge"])
        ax.set_title(setting.replace("ctt_", ""))
        ax.set_xscale("log")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="x", visible=True, which="both", linewidth=0.5)
    axes[0, 0].set_ylabel("Recall")
    axes[1, 0].set_ylabel("Recall")
    axes[1, 0].set_xlabel("FPR budget")
    axes[1, 1].set_xlabel("FPR budget")
    legend_for_models(axes[1, 1], y=-0.35)
    add_header(fig, "Recall under measured low-FPR budgets", "Only available budgets are plotted: 1e-4, 5e-4 and 1e-3; no extrapolated budgets are drawn.")
    fig.subplots_adjust(top=0.78, bottom=0.22, hspace=0.32, wspace=0.12)
    savefig(fig, "paper_fig7_recall_at_fpr")
    return out


def fig_efficiency_tradeoff(missing: list[str]) -> pd.DataFrame:
    eff = pd.read_csv(TABLE_DIR / "efficiency_road.csv")
    main = pd.read_csv(TABLE_DIR / "road_main_20ep.csv")
    wanted = ["cnn", "lstm", "gru", "transformer", "concat_fusion", "cmf_can"]
    rows = []
    for model in wanted:
        e = eff[eff["model"] == model]
        r = main[main["model"] == model]
        if e.empty:
            missing.append(f"paper_fig8_efficiency_tradeoff: missing efficiency for {MODEL_LABELS.get(model, model)}")
        rows.append(
            {
                "Model": MODEL_LABELS.get(model, model),
                "latency_batch_ms": np.nan if e.empty else e["avg_batch_ms"].iloc[0],
                "throughput_windows_per_s": np.nan if e.empty else e["throughput_windows_per_s"].iloc[0],
                "params": np.nan if e.empty else e["total_params"].iloc[0],
                "F1": np.nan if r.empty else r["f1"].iloc[0],
                "AUPR": np.nan if r.empty else r["aupr"].iloc[0],
            }
        )
    out = pd.DataFrame(rows)
    out["latency_window_us"] = out["latency_batch_ms"] * 1000 / 512
    out.to_csv(TABLE_DIR / "paper_table_efficiency_tradeoff.csv", index=False)
    out.to_csv(TABLE_DIR / "paper_table_efficiency_summary.csv", index=False)
    tex = out.copy()
    for c in ["latency_batch_ms", "throughput_windows_per_s", "params", "F1", "AUPR", "latency_window_us"]:
        tex[c] = tex[c].map(lambda x: fmt_num(x, 4))
    write_tex(tex, TABLE_DIR / "paper_table_efficiency_summary.tex")

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    colors = {"CNN": "#C5CAD3", "LSTM": "#E2E5EA", "GRU": "#B8A037", "Transformer": "#A3BEFA", "Concat-Fusion": "#FFE15B", "CMF-CAN": "#F0986E"}
    markers = {"CNN": "o", "LSTM": "s", "GRU": "D", "Transformer": "o", "Concat-Fusion": "s", "CMF-CAN": "^"}
    for ax, y, label in [(axes[0], "F1", "F1"), (axes[1], "AUPR", "AUPR")]:
        for rec in out.itertuples():
            if pd.isna(rec.latency_window_us) or pd.isna(getattr(rec, y)):
                continue
            ax.scatter(
                rec.latency_window_us,
                getattr(rec, y),
                s=52 if rec.Model == "CMF-CAN" else 42,
                color=colors[rec.Model],
                marker=markers[rec.Model],
                edgecolor="#464C55",
                linewidth=1.15 if rec.Model == "CMF-CAN" else 0.8,
                label=rec.Model,
            )
        ax.set_xlabel("Latency per window (us)")
        ax.set_ylabel(label)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="x", visible=True)
    handles, labels = axes[0].get_legend_handles_labels()
    seen = {}
    for h, lab in zip(handles, labels):
        seen.setdefault(lab, h)
    axes[1].legend(seen.values(), seen.keys(), frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(-0.1, -0.34), columnspacing=1.2)
    add_header(fig, "Efficiency-performance trade-off on ROAD", "Scatter uses measured CUDA latency and ROAD main-result F1/AUPR; no missing efficiency values are imputed.")
    fig.subplots_adjust(top=0.72, bottom=0.38, wspace=0.28)
    savefig(fig, "paper_fig8_efficiency_tradeoff")
    return out


def dataset_summary(missing: list[str]) -> pd.DataFrame:
    usage = {
        "road": "Main, few-label, ablation, efficiency",
        "ctt_test01": "Main, few-label, generalization, ablation, low-FPR",
        "ctt_test02": "Known attack / unknown vehicle generalization",
        "ctt_test03": "Unknown attack / known vehicle generalization",
        "ctt_test04": "Unknown vehicle / unknown attack generalization",
        "hcrl_can_intrusion": "Optional main result",
        "car_hacking": "Optional main result",
        "crysys_family_mod_subset": "Optional main result",
    }
    rows = []
    for ds, use in usage.items():
        root = DATA_DIR / ds
        meta_path = next(root.glob("*_prepare_meta.json"), None) if root.exists() else None
        meta = json.loads(meta_path.read_text()) if meta_path else {}
        parquet = root / "frames.parquet"
        windows = root / "windows_index.npy"
        frames = windows_n = attack_ratio = vehicles = np.nan
        attack_types = "NA"
        if parquet.exists():
            try:
                frame_df = pd.read_parquet(parquet, columns=["label", "attack_type", "vehicle"])
                frames = len(frame_df)
                attack_ratio = frame_df["label"].mean()
                vehicles = frame_df["vehicle"].nunique()
                atks = sorted([str(x) for x in frame_df["attack_type"].dropna().unique() if str(x) != "normal"])
                attack_types = ", ".join(atks[:8]) + (" ..." if len(atks) > 8 else "")
            except Exception as exc:  # pragma: no cover - diagnostic path
                missing.append(f"paper_table_dataset_summary: could not read {parquet}: {exc}")
        else:
            missing.append(f"paper_table_dataset_summary: missing frames parquet for {ds}")
        if windows.exists():
            windows_n = int(np.load(windows, mmap_mode="r").shape[0])
        else:
            missing.append(f"paper_table_dataset_summary: missing windows_index for {ds}")
        split_counts = meta.get("split_counts", {})
        rows.append(
            {
                "Dataset": ds,
                "Usage": use,
                "Split Protocol": meta.get("split_method") or meta.get("source_test_folder") or "time/group split",
                "Vehicles": vehicles,
                "Attack Types": attack_types,
                "Train/Val/Test Setting": f"{split_counts.get('0', 'NA')}/{split_counts.get('1', 'NA')}/{split_counts.get('2', 'NA')}",
                "Frames": frames,
                "Windows": windows_n,
                "Attack Ratio": attack_ratio,
                "Notes": meta.get("label_note") or meta.get("frame_label_note") or "NA",
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "paper_table_dataset_summary.csv", index=False)
    tex = out.copy()
    for c in ["Vehicles", "Frames", "Windows"]:
        tex[c] = tex[c].map(lambda x: "NA" if pd.isna(x) else f"{int(x):,}")
    tex["Attack Ratio"] = tex["Attack Ratio"].map(lambda x: "NA" if pd.isna(x) else f"{x:.2%}")
    write_tex(tex, TABLE_DIR / "paper_table_dataset_summary.tex")
    return out


def appendix_summaries(missing: list[str]) -> None:
    for name, reason in [
        ("PR/ROC curves", "requires per-sample score/probability and label"),
        ("Per-attack recall/F1", "requires per-sample prediction, label and attack_type"),
        ("Gate weights by attack/setting", "requires saved gate_frame, gate_window and gate_context"),
        ("UMAP/t-SNE representation", "requires embedding dump"),
        ("Failure case analysis", "requires per-sample prediction"),
    ]:
        missing.append(f"{name}: not generated because it {reason}.")
    if (TABLE_DIR / "calibration_summary.csv").exists():
        cal = pd.read_csv(TABLE_DIR / "calibration_summary.csv")
        cal.to_csv(TABLE_DIR / "paper_table_calibration_summary.csv", index=False)
        tex = cal.copy()
        num_cols = tex.select_dtypes(include=[np.number]).columns
        for c in num_cols:
            tex[c] = tex[c].map(lambda x: fmt_num(x, 4))
        write_tex(tex, TABLE_DIR / "paper_table_calibration_summary.tex")
        missing.append("Calibration curve/reliability diagram: bin-level calibration data is missing; generated calibration summary table only.")
    else:
        missing.append("Calibration curve/reliability diagram: calibration_summary.csv missing.")

    if (TABLE_DIR / "ood_score_summary.csv").exists():
        ood = pd.read_csv(TABLE_DIR / "ood_score_summary.csv")
        ood.to_csv(TABLE_DIR / "paper_table_ood_score_summary.csv", index=False)
        fig, ax = plt.subplots(figsize=(5.2, 2.6))
        plot = ood.copy()
        plot["Setting"] = plot["dataset"].str.replace("ctt_", "", regex=False)
        sns.barplot(data=plot, x="Setting", y="f1", ax=ax, color="#A3BEFA", edgecolor="#2E4780", linewidth=0.8)
        ax.set_ylabel("F1")
        ax.set_xlabel("")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="x", visible=False)
        add_header(fig, "OOD score summary", "Summary-level OOD scores only; no distribution is plotted without per-sample scores.", y=1.02)
        fig.subplots_adjust(top=0.72, bottom=0.18)
        savefig(fig, "paper_fig_appendix_ood_score_summary")
        missing.append("OOD score distribution: per-sample OOD scores are missing; generated summary bar/table only.")
    else:
        missing.append("OOD score distribution: ood_score_summary.csv missing.")


def missing_report(missing: list[str]) -> None:
    lines = [
        "# Missing Inputs Report",
        "",
        "This report is generated from the current VIDS workspace and only records unavailable evidence. No unsupported figures were generated.",
        "",
        "## Missing or Partial Inputs",
        *[f"- {item}" for item in missing],
        "",
        "## Required Fields for Full Appendix Coverage",
        "- sample_id",
        "- dataset",
        "- setting",
        "- model",
        "- label",
        "- pred",
        "- score",
        "- attack_type",
        "- vehicle",
        "- gate_frame",
        "- gate_window",
        "- gate_context",
        "- embedding_path",
        "",
        "## Suggested Script Changes",
        "- Add an evaluation export option in `cmf_can/training/cli.py` or `cmf_can/training/train.py` to save per-window predictions with the fields above.",
        "- Add a CMF-CAN forward/evaluation hook in `cmf_can/models/cmf.py` / evaluation code to persist gate weights per sample.",
        "- Add an embedding dump option before the classifier head for UMAP/t-SNE and failure-case analysis.",
        "- Save calibration bin statistics if reliability diagrams are required.",
    ]
    (TABLE_DIR / "missing_inputs_report.md").write_text("\n".join(lines) + "\n")


def inventory() -> None:
    fig_entries = [
        ("paper_fig1_architecture", "results/cmf_figures/fig_model_architecture_cmf_can.*", "正文", "Model architecture."),
        ("paper_fig2_main_multidataset", "road_main_20ep.csv, ctt_generalization_15ep.csv, hcrl_main_15ep.csv, car_hacking_main_15ep.csv, crysys_family_mod_3model_3seed_mean_std.csv", "正文", "Main multi-dataset performance."),
        ("paper_fig3_road_few_label", "road_few_label_3seed_mean_std.csv", "正文", "ROAD few-label curve."),
        ("paper_fig4_ctt_few_label", "ctt_few_label_3seed_mean_std.csv", "正文", "CT&T few-label curve."),
        ("paper_fig5_ctt_generalization", "ctt_generalization_15ep.csv", "正文", "Known/unknown generalization."),
        ("paper_fig6_ablation", "road_ablation_20ep_merged.csv, ctt_ablation_15ep_merged.csv", "正文", "Ablation study."),
        ("paper_fig7_recall_at_fpr", "ctt_deployment_low_fpr_15ep.csv", "正文/附录", "Low-FPR deployment recall."),
        ("paper_fig8_efficiency_tradeoff", "efficiency_road.csv, road_main_20ep.csv", "正文/附录", "Efficiency-performance trade-off."),
        ("paper_fig_appendix_ood_score_summary", "ood_score_summary.csv", "附录", "Summary bar only; no distribution without per-sample scores."),
    ]
    table_entries = [
        "paper_table_dataset_summary",
        "paper_table_overall_main_results",
        "paper_table_few_label_summary",
        "paper_table_ctt_generalization_summary",
        "paper_table_low_fpr_summary",
        "paper_table_efficiency_summary",
        "paper_table_calibration_summary",
        "paper_table_ood_score_summary",
    ]
    lines = ["# Paper Figure/Table Inventory", "", "## Generated Figures"]
    lines += [f"- `{name}.png/.pdf/.svg`: input `{src}`; placement: {place}; note: {note}" for name, src, place, note in fig_entries if (FIG_DIR / f"{name}.png").exists() or (FIG_DIR / f"{name}.svg").exists()]
    lines += ["", "## Generated Tables"]
    lines += [f"- `{name}.csv/.tex`" if (TABLE_DIR / f"{name}.tex").exists() else f"- `{name}.csv`" for name in table_entries if (TABLE_DIR / f"{name}.csv").exists()]
    lines += [
        "",
        "## Not Generated Because Inputs Are Missing",
        "- PR/ROC curves: no per-sample score/probability and label dump.",
        "- Per-attack recall/F1: no per-sample prediction with attack_type.",
        "- Gate weights by attack/setting: no saved gate weights.",
        "- UMAP/t-SNE: no embedding dump.",
        "- Failure case analysis: no per-sample prediction dump.",
        "- Reliability curve: no bin-level calibration data.",
        "- OOD score distribution: no per-sample OOD scores.",
    ]
    (TABLE_DIR / "paper_figure_table_inventory.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    os.chdir(args.root.resolve())
    set_style()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    missing: list[str] = []
    copy_architecture(missing)
    dataset_summary(missing)
    main_results = collect_main_results()
    fig_main_multidataset(main_results)
    fig_and_table_few_label()
    fig_ctt_generalization()
    fig_ablation(missing)
    fig_recall_at_fpr(missing)
    fig_efficiency_tradeoff(missing)
    appendix_summaries(missing)
    missing_report(missing)
    inventory()
    print("[write] paper-ready figures and tables")


if __name__ == "__main__":
    main()
