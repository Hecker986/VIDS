from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


TABLE_DIR = Path("results/cmf_tables")
FIG_DIR = Path("results/cmf_figures")
MODEL_ORDER = ["transformer", "concat_fusion", "cmf_can"]
MODEL_LABELS = {
    "transformer": "Transformer",
    "concat_fusion": "Concat",
    "cmf_can": "CMF-CAN",
}
DATASET_LABELS = {
    "road": "ROAD",
    "ctt_test01": "CT&T test01",
    "hcrl_can_intrusion": "HCRL",
    "car_hacking": "Car-Hacking",
    "crysys_family_mod_subset": "CrySyS",
}


def set_style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.titlesize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": "#E5E7EB",
            "grid.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def savefig(fig: mpl.figure.Figure, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "pdf", "png"):
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.04}
        if ext == "png":
            kwargs["dpi"] = 300
        fig.savefig(FIG_DIR / f"{stem}.{ext}", **kwargs)
    plt.close(fig)


def shifted_unknown() -> None:
    df = pd.read_csv(TABLE_DIR / "anomaly_ensemble_final_shifted_summary.csv")
    rows = []
    for rec in df.itertuples():
        short = rec.dataset.replace("ctt_", "")
        rows.extend(
            [
                {"dataset": short, "metric": "F1", "method": "Baseline", "value": rec.baseline_f1_mean, "std": rec.baseline_f1_std},
                {"dataset": short, "metric": "F1", "method": "Enhanced", "value": rec.best_f1_mean, "std": rec.best_f1_std},
                {"dataset": short, "metric": "AUPR", "method": "Baseline", "value": rec.baseline_aupr_mean, "std": rec.baseline_aupr_std},
                {"dataset": short, "metric": "AUPR", "method": "Enhanced", "value": rec.best_f1_aupr_mean, "std": rec.best_f1_aupr_std},
                {
                    "dataset": short,
                    "metric": "Recall@FPR<=1e-4",
                    "method": "Baseline",
                    "value": rec.baseline_recall_at_fpr_1em04_mean,
                    "std": rec.baseline_recall_at_fpr_1em04_std,
                },
                {
                    "dataset": short,
                    "metric": "Recall@FPR<=1e-4",
                    "method": "Enhanced",
                    "value": rec.best_lowfpr_recall_at_fpr_1em04_mean,
                    "std": rec.best_lowfpr_recall_at_fpr_1em04_std,
                },
            ]
        )
    plot = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35), sharey=True)
    palette = {"Baseline": "#B8C2CC", "Enhanced": "#3B73C6"}
    hatches = {"Baseline": "///", "Enhanced": ""}
    for ax, metric in zip(axes, ["F1", "AUPR", "Recall@FPR<=1e-4"]):
        sub = plot[plot["metric"] == metric]
        sns.barplot(data=sub, x="dataset", y="value", hue="method", palette=palette, ax=ax, errorbar=None, edgecolor="#263238", linewidth=0.4)
        for patch, (_, row) in zip(ax.patches, sub.sort_values(["dataset", "method"]).iterrows()):
            patch.set_hatch(hatches[row["method"]])
        for i, dataset in enumerate(sub["dataset"].unique()):
            for j, method in enumerate(["Baseline", "Enhanced"]):
                r = sub[(sub.dataset == dataset) & (sub.method == method)].iloc[0]
                x = i + (-0.2 if method == "Baseline" else 0.2)
                ax.errorbar(x, r.value, yerr=r["std"], fmt="none", ecolor="#334155", elinewidth=0.7, capsize=2)
        ax.set_title(metric)
        ax.set_xlabel("")
        ax.set_ylabel("Score" if ax is axes[0] else "")
        ax.set_ylim(0, 1.02)
        if ax.legend_:
            ax.legend_.remove()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.53, 1.08))
    fig.suptitle("Shifted / Unknown Attack Repair", y=1.18, fontweight="bold")
    savefig(fig, "mpl_shifted_unknown_final")


def road_label_ratio() -> None:
    df = pd.read_csv(TABLE_DIR / "label_ratio_coverage_summary.csv")
    road = df[df.dataset == "road"].sort_values("label_ratio")
    x = np.arange(len(road))
    labels = [f"{v:g}" for v in road.label_ratio]
    fig, ax = plt.subplots(figsize=(5.2, 2.7))
    series = [
        ("CMF F1", "cmf_can_f1_mean", "#3B73C6", "o"),
        ("CMF AUPR", "cmf_can_aupr_mean", "#2F9A61", "s"),
        ("CMF Recall@FPR<=1e-4", "cmf_can_recall_at_fpr_1em04_mean", "#D9812B", "^"),
    ]
    for name, col, color, marker in series:
        ax.plot(x, road[col], color=color, marker=marker, linewidth=1.8, markersize=4.5, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Label ratio")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.02)
    ax.set_title("ROAD few-label behavior", fontweight="bold")
    ax.legend(frameon=False, ncol=1, loc="lower right")
    savefig(fig, "mpl_road_label_ratio_metrics")


def test04_progression() -> None:
    vals = pd.DataFrame(
        [
            {"method": "CMF-CAN", "F1": 0.0284, "AUPR": 0.2210, "Recall@FPR<=1e-4": 0.0148},
            {"method": "+Anomaly", "F1": 0.2412, "AUPR": 0.2992, "Recall@FPR<=1e-4": 0.1329},
            {"method": "+Temporal", "F1": 0.4457, "AUPR": 0.2992, "Recall@FPR<=1e-4": 0.1329},
        ]
    )
    long = vals.melt("method", var_name="metric", value_name="score")
    fig, ax = plt.subplots(figsize=(5.4, 2.7))
    sns.barplot(data=long, x="method", y="score", hue="metric", ax=ax, palette=["#3B73C6", "#2F9A61", "#D9812B"], edgecolor="#263238", linewidth=0.4)
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 0.55)
    ax.set_title("CT&T test04 repair progression", fontweight="bold")
    ax.legend(frameon=False, ncol=1, loc="upper left")
    savefig(fig, "mpl_test04_progression")


def _append_single_seed_rows(rows: list[dict], path: Path, dataset: str) -> None:
    df = pd.read_csv(path)
    df = df[df["model"].isin(MODEL_ORDER)]
    for rec in df.itertuples():
        rows.append(
            {
                "dataset": DATASET_LABELS[dataset],
                "model": MODEL_LABELS[rec.model],
                "f1_mean": rec.f1,
                "f1_std": 0.0,
                "recall_lowfpr_mean": rec.recall_at_fpr_1em04,
                "recall_lowfpr_std": 0.0,
            }
        )


def model_comparison_summary() -> pd.DataFrame:
    rows: list[dict] = []

    road = pd.read_csv(TABLE_DIR / "road_few_label_3seed_mean_std.csv")
    road = road[(road["label_ratio"] == 1.0) & road["model"].isin(MODEL_ORDER)]
    for rec in road.itertuples():
        rows.append(
            {
                "dataset": DATASET_LABELS["road"],
                "model": MODEL_LABELS[rec.model],
                "f1_mean": rec.f1_mean,
                "f1_std": rec.f1_std,
                "recall_lowfpr_mean": rec.recall_at_fpr_1em04_mean,
                "recall_lowfpr_std": rec.recall_at_fpr_1em04_std,
            }
        )

    ctt = pd.read_csv(TABLE_DIR / "ctt_few_label_3seed_mean_std.csv")
    ctt = ctt[(ctt["dataset"] == "ctt_test01") & (ctt["label_ratio"] == 1.0) & ctt["model"].isin(MODEL_ORDER)]
    for rec in ctt.itertuples():
        rows.append(
            {
                "dataset": DATASET_LABELS["ctt_test01"],
                "model": MODEL_LABELS[rec.model],
                "f1_mean": rec.f1_mean,
                "f1_std": rec.f1_std,
                "recall_lowfpr_mean": rec.recall_at_fpr_1em04_mean,
                "recall_lowfpr_std": rec.recall_at_fpr_1em04_std,
            }
        )

    _append_single_seed_rows(rows, TABLE_DIR / "hcrl_main_15ep.csv", "hcrl_can_intrusion")
    _append_single_seed_rows(rows, TABLE_DIR / "car_hacking_main_15ep.csv", "car_hacking")

    cry = pd.read_csv(TABLE_DIR / "crysys_family_mod_3model_3seed_mean_std.csv")
    piv = cry[cry["model"].isin(MODEL_ORDER)].pivot_table(index="model", columns="metric", values=["mean", "std"], aggfunc="first")
    for model in MODEL_ORDER:
        rows.append(
            {
                "dataset": DATASET_LABELS["crysys_family_mod_subset"],
                "model": MODEL_LABELS[model],
                "f1_mean": piv.loc[model, ("mean", "f1")],
                "f1_std": piv.loc[model, ("std", "f1")],
                "recall_lowfpr_mean": piv.loc[model, ("mean", "recall_at_fpr_1em04")],
                "recall_lowfpr_std": piv.loc[model, ("std", "recall_at_fpr_1em04")],
            }
        )

    out = pd.DataFrame(rows)
    out["dataset"] = pd.Categorical(out["dataset"], categories=list(DATASET_LABELS.values()), ordered=True)
    out["model"] = pd.Categorical(out["model"], categories=[MODEL_LABELS[m] for m in MODEL_ORDER], ordered=True)
    return out.sort_values(["dataset", "model"])


def _plot_model_metric(df: pd.DataFrame, metric: str, yerr: str, ylabel: str, title: str, subtitle: str, stem: str) -> None:
    datasets = list(df["dataset"].cat.categories)
    models = [MODEL_LABELS[m] for m in MODEL_ORDER]
    x = np.arange(len(datasets))
    width = 0.24
    colors = {"Transformer": "#A3BEFA", "Concat": "#FFE15B", "CMF-CAN": "#F0986E"}
    hatches = {"Transformer": "///", "Concat": "\\\\\\", "CMF-CAN": ""}
    fig, ax = plt.subplots(figsize=(7.3, 3.05))
    for j, model in enumerate(models):
        sub = df[df["model"] == model].set_index("dataset").reindex(datasets)
        pos = x + (j - 1) * width
        bars = ax.bar(
            pos,
            sub[metric].to_numpy(),
            width=width,
            label=model,
            color=colors[model],
            edgecolor="#263238",
            linewidth=0.45,
            hatch=hatches[model],
        )
        ax.errorbar(
            pos,
            sub[metric].to_numpy(),
            yerr=sub[yerr].fillna(0).to_numpy(),
            fmt="none",
            ecolor="#334155",
            elinewidth=0.75,
            capsize=2,
            capthick=0.75,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=18, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.set_ylim(0, 1.05)
    ax.set_title(title, fontweight="bold", loc="left", pad=20)
    ax.text(0.0, 1.025, subtitle, transform=ax.transAxes, ha="left", va="bottom", fontsize=8, color="#6F768A")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.24), handlelength=2.8, columnspacing=1.8)
    ax.grid(axis="y", linestyle="-", linewidth=0.8, color="#E6E8F0")
    ax.grid(axis="x", visible=False)
    fig.subplots_adjust(bottom=0.29, top=0.8)
    savefig(fig, stem)


def model_comparison() -> None:
    df = model_comparison_summary()
    _plot_model_metric(
        df,
        metric="f1_mean",
        yerr="f1_std",
        ylabel="F1",
        title="Model comparison across evaluated datasets",
        subtitle="Mean F1 at full-label setting; error bars show standard deviation when multi-seed results are available.",
        stem="mpl_model_comparison_f1",
    )
    _plot_model_metric(
        df,
        metric="recall_lowfpr_mean",
        yerr="recall_lowfpr_std",
        ylabel="Recall@FPR<=1e-4",
        title="Deployment-oriented low-false-positive comparison",
        subtitle="Recall at FPR<=1e-4; ROAD, CT&T and CrySyS use multi-seed summaries, HCRL/Car-Hacking are single-seed.",
        stem="mpl_model_comparison_low_fpr",
    )
    df.to_csv(TABLE_DIR / "model_comparison_summary.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    os.chdir(args.root.resolve())
    set_style()
    shifted_unknown()
    road_label_ratio()
    test04_progression()
    model_comparison()
    print("[write] matplotlib paper figures")


if __name__ == "__main__":
    main()
