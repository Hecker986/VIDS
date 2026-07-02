from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd


FINAL_SUPP = Path("results/final_paper_supplement")
ATTACK = Path("results/attack_centric_final")
SUPP_FIGS = FINAL_SUPP / "figures"
ATTACK_FIGS = ATTACK / "figures"

RED = "#D00000"
RED_DARK = "#8B0000"
GREY_LIGHT = "#D3D3D3"
GREY_MID = "#A9A9A9"
GREY_DARK = "#707070"
ORANGE_LIGHT = "#FFB695"
ORANGE_MID = "#FF7F5E"
BLUE = "#4C78A8"
GREEN = "#1B9E77"


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "STIX Two Text", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.unicode_minus": False,
            "hatch.color": "white",
            "hatch.linewidth": 1.35,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def polish_axis(ax, ylabel: str, ylim: tuple[float, float] = (0, 1.0)) -> None:
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel, fontsize=10.8)
    ax.yaxis.grid(True, color="#EBEBEB", linewidth=0.75, linestyle="--", zorder=0)
    ax.set_axisbelow(True)
    for side, spine in ax.spines.items():
        if side in ("top", "right"):
            spine.set_visible(False)
        else:
            spine.set_linewidth(0.9)
            spine.set_color("#333333")
    ax.tick_params(axis="both", length=0, labelsize=9.8)


def add_value_labels(ax, bars, values, is_best: bool = False, dy: float = 0.012) -> None:
    for bar, val in zip(bars, values):
        if not np.isfinite(val):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + dy,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=8.1,
            fontweight="bold" if is_best else "normal",
            color=RED_DARK if is_best else "#222222",
            zorder=5,
        )


def grouped_bar(
    path_base: Path,
    groups: list[str],
    data: dict[str, list[float]],
    title: str,
    ylabel: str,
    best_method: str,
    colors: dict[str, str],
    ylim: tuple[float, float] = (0, 1.05),
    legend_anchor: tuple[float, float] = (0.995, 0.99),
    legend_outside: bool = True,
) -> None:
    labels = list(data.keys())
    x = np.arange(len(groups), dtype=float)
    n = len(labels)
    total_w = 0.78
    bar_w = total_w / n

    fig, ax = plt.subplots(figsize=(7.45, 3.25))
    for i, label in enumerate(labels):
        vals = np.asarray(data[label], dtype=float)
        offset = (i - n / 2 + 0.5) * bar_w
        is_best = label == best_method
        bars = ax.bar(
            x + offset,
            vals,
            width=bar_w,
            color=colors.get(label, GREY_MID),
            hatch="//" if is_best else "",
            edgecolor="white",
            linewidth=0.85,
            zorder=3,
            label=label,
        )
        add_value_labels(ax, bars, vals, is_best=is_best, dy=(ylim[1] - ylim[0]) * 0.012)

    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=9.8)
    ax.set_title(title, fontsize=12.2, pad=7, fontweight="normal")
    polish_axis(ax, ylabel, ylim)
    ax.set_xlim(-0.55, len(groups) - 0.45)

    handles = [
        mpatches.Patch(
            facecolor=colors.get(label, GREY_MID),
            hatch="//" if label == best_method else "",
            edgecolor="white",
            linewidth=0.85,
            label=label,
        )
        for label in labels
    ]
    if legend_outside:
        loc = "lower center"
        bbox = (0.5, 1.02)
        ncol = min(len(labels), 4)
    else:
        loc = "upper right"
        bbox = legend_anchor
        ncol = 1
    leg = ax.legend(
        handles=handles,
        loc=loc,
        bbox_to_anchor=bbox,
        ncol=ncol,
        fontsize=8.3,
        framealpha=1.0,
        facecolor="white",
        edgecolor="#C8C8C8",
        fancybox=False,
        borderpad=0.28,
        labelspacing=0.26,
        handlelength=1.65,
        handletextpad=0.45,
        borderaxespad=0.25,
    )
    for text in leg.get_texts():
        if text.get_text() == best_method:
            text.set_fontweight("bold")

    fig.tight_layout(pad=0.55)
    fig.savefig(path_base.with_suffix(".svg"), format="svg")
    fig.savefig(path_base.with_suffix(".pdf"), format="pdf")
    plt.close(fig)


def completed_low_fpr() -> None:
    df = pd.read_csv(ATTACK / "tables/completed_test04_low_fpr_event.csv")
    keep = [
        "GRAIN window100",
        "SAFE-CAN GradientBoosting",
        "CAN-Transformer+ same-ID",
        "CMF-CAN",
    ]
    df = df[df["model"].isin(keep)].set_index("model")
    groups = ["AUPR", "R@1e-4", "R@1e-3", "Event"]
    metric_cols = ["aupr", "recall_at_fpr_0_0001", "recall_at_fpr_0_001", "event_recall"]
    data = {model: [float(df.loc[model, c]) if pd.notna(df.loc[model, c]) else 0.0 for c in metric_cols] for model in keep}
    colors = {
        "Predict all normal": GREY_LIGHT,
        "Old window100 Transformer": GREY_MID,
        "CMF-CAN": GREY_DARK,
        "CAN-Transformer+ same-ID": BLUE,
        "SAFE-CAN GradientBoosting": ORANGE_MID,
        "GRAIN window100": RED,
    }
    grouped_bar(
        ATTACK_FIGS / "paper_fig10_completed_low_fpr_event",
        groups,
        data,
        "(a) CT\\&T Test04 Low-FPR Evidence",
        "Score / recall",
        "GRAIN window100",
        colors,
        ylim=(0, 1.08),
    )


def corrected_benchmark() -> None:
    df = pd.read_csv(FINAL_SUPP / "tables/main_ctt_corrected_benchmark.csv")
    settings = ["ctt_test01", "ctt_test02", "ctt_test03", "ctt_test04"]
    rows = [
        ("Table13 GB", "Table13-style GradientBoosting"),
        ("Old Transformer", "old window100 Transformer"),
        ("CMF-CAN", "CMF-CAN"),
        ("GRAIN best", None),
    ]

    mat = np.full((len(rows), len(settings)), np.nan, dtype=float)
    for j, setting in enumerate(settings):
        sub = df[df["setting"].eq(setting)]
        for i, (_, model) in enumerate(rows):
            if model is None:
                vals = pd.to_numeric(
                    sub[sub["model"].astype(str).str.startswith("GRAIN_window")]["attack_f1"],
                    errors="coerce",
                )
                val = vals.max()
            else:
                vals = pd.to_numeric(sub[sub["model"].eq(model)]["attack_f1"], errors="coerce")
                val = vals.iloc[0] if len(vals) else np.nan
            mat[i, j] = float(val) if pd.notna(val) else np.nan

    cmap = LinearSegmentedColormap.from_list(
        "attack_f1_spice",
        ["#F7F7F7", "#FDE5D3", "#FFB695", "#FF7F5E", RED],
    )
    masked = np.ma.masked_invalid(mat)

    fig, ax = plt.subplots(figsize=(7.45, 2.85))
    im = ax.imshow(masked, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")

    ax.set_title("(b) Corrected Attack-F1 Across CT\\&T Settings", fontsize=12.0, pad=8, fontweight="normal")
    ax.set_xticks(np.arange(len(settings)))
    ax.set_xticklabels(["T01", "T02", "T03", "T04"], fontsize=9.8)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=9.2)
    ax.tick_params(axis="both", length=0)

    for i in range(len(rows)):
        for j in range(len(settings)):
            val = mat[i, j]
            if np.isfinite(val):
                color = "white" if val >= 0.72 else "#222222"
                weight = "bold" if rows[i][0] == "GRAIN best" else "normal"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8.6, color=color, fontweight=weight)
            else:
                ax.text(j, i, "N/A", ha="center", va="center", fontsize=8.3, color="#555555", fontstyle="italic")

    ax.set_xticks(np.arange(-0.5, len(settings), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    for side, spine in ax.spines.items():
        spine.set_visible(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.028, pad=0.025)
    cbar.set_label("Attack-F1", fontsize=9.2)
    cbar.ax.tick_params(labelsize=8.2, length=0)
    cbar.outline.set_visible(False)

    ax.text(
        -0.48,
        len(rows) - 0.08,
        "Values are corrected attack-positive F1; unavailable entries are not drawn as zero.",
        ha="left",
        va="top",
        fontsize=7.7,
        color="#444444",
        transform=ax.transData,
    )

    fig.tight_layout(pad=0.55)
    path_base = SUPP_FIGS / "main_ctt_corrected_benchmark"
    fig.savefig(path_base.with_suffix(".svg"), format="svg")
    fig.savefig(path_base.with_suffix(".pdf"), format="pdf")
    plt.close(fig)


def grain_ablation() -> None:
    df = pd.read_csv(FINAL_SUPP / "tables/main_grain_ablation.csv")
    labels = ["sample-level", "window_10", "window_20", "window_100 aggregate", "old window100 deep Transformer"]
    short = ["Sample", "W10", "W20", "W100", "Old-T"]
    data = {
        "Attack-F1": [float(df[df["ablation"].eq(x)]["attack_f1"].iloc[0]) for x in labels],
        "AUPR": [float(df[df["ablation"].eq(x)]["aupr"].iloc[0]) for x in labels],
    }
    colors = {"Attack-F1": ORANGE_MID, "AUPR": RED}
    grouped_bar(
        SUPP_FIGS / "main_grain_ablation",
        short,
        data,
        "(c) Feature-Preserving Granularity",
        "Metric value",
        "AUPR",
        colors,
        ylim=(0, 0.9),
        legend_anchor=(0.995, 0.995),
    )


def ranking_inversion() -> None:
    df = pd.read_csv(FINAL_SUPP / "tables/main_ranking_inversion.csv")
    groups = ["T01", "T02", "T03", "T04"]
    data = {
        "Top-3 overlap": df["top3_overlap_weighted_vs_attack"].astype(float).tolist(),
        "All-normal rank/10": (df["predict_all_normal_rank_by_weighted"].astype(float) / 10.0).tolist(),
    }
    colors = {"Top-3 overlap": RED, "All-normal rank/10": GREY_MID}
    grouped_bar(
        SUPP_FIGS / "main_ranking_inversion",
        groups,
        data,
        "(d) Ranking Inversion Diagnostics",
        "Normalized value",
        "Top-3 overlap",
        colors,
        ylim=(0, 1.05),
    )


def main() -> None:
    setup_style()
    completed_low_fpr()
    corrected_benchmark()
    grain_ablation()
    ranking_inversion()


if __name__ == "__main__":
    main()
