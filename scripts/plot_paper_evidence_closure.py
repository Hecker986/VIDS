from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUT = Path("results/paper_revision_strict_review/figures")
TABLES = Path("results/final_evidence_completion/tables")


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "STIX Two Text", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 8.5,
            "axes.linewidth": 0.8,
            "hatch.linewidth": 0.9,
        }
    )


def open_axis(ax) -> None:
    ax.grid(axis="y", color="#E6E6E6", linestyle="--", linewidth=0.65, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", length=0, labelsize=8)


def main() -> None:
    style()
    OUT.mkdir(parents=True, exist_ok=True)
    low = pd.read_csv(TABLES / "formal_validation_low_fpr.csv")
    feat = pd.read_csv(TABLES / "grain_multiseed_feature_removal_summary.csv")
    status = pd.read_csv(TABLES / "final_material_experiment_completion_status.csv")

    low_rows = low[low["threshold_type"].eq("validation_fpr_budget_0.0005")].copy()
    order = ["ctt_test01", "ctt_test02", "ctt_test03", "ctt_test04"]
    low_rows["setting"] = pd.Categorical(low_rows["setting"], order, ordered=True)
    low_rows = low_rows.sort_values("setting")

    feat_order = [
        "only_timing",
        "full_safe_can",
        "without_delta_t_same_id",
        "only_payload",
        "only_id",
    ]
    feat = feat[feat["ablation"].isin(feat_order)].copy()
    feat["ablation"] = pd.Categorical(feat["ablation"], feat_order, ordered=True)
    feat = feat.sort_values("ablation")

    status_counts = status["status"].value_counts()
    status_labels = ["completed", "audited_external_dependency", "protocol_marker_retained"]
    status_values = [int(status_counts.get(x, 0)) for x in status_labels]
    status_text = ["Completed", "Audited external", "Protocol marker"]

    fig, axes = plt.subplots(1, 3, figsize=(7.45, 2.42), gridspec_kw={"wspace": 0.46})

    x = np.arange(len(low_rows))
    axes[0].bar(
        x,
        low_rows["recall"],
        color="#D9D9D9",
        edgecolor="black",
        hatch="///",
        linewidth=0.8,
        width=0.62,
        zorder=3,
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(["T01", "T02", "T03", "T04"])
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Recall")
    axes[0].set_title("(a) Val-threshold low-FPR", fontsize=9.2)
    for xi, val in zip(x, low_rows["recall"]):
        axes[0].text(xi, val + 0.035, f"{val:.2f}", ha="center", fontsize=7.2)
    open_axis(axes[0])

    x2 = np.arange(len(feat))
    colors = ["#F2F2F2", "#CFE3F5", "#F4CCCC", "#E6E6E6", "#E6E6E6"]
    hatches = ["//", "", "xx", "..", "\\\\"]
    bars = axes[1].bar(
        x2,
        feat["attack_f1_mean"],
        yerr=feat["attack_f1_std"].fillna(0),
        color=colors,
        edgecolor="black",
        linewidth=0.8,
        width=0.62,
        capsize=2,
        zorder=3,
    )
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(["Timing\nonly", "Full", "$-\\Delta t$", "Payload\nonly", "ID\nonly"], fontsize=7.3)
    axes[1].set_ylim(0, 0.72)
    axes[1].set_ylabel("$F1_{attack}$")
    axes[1].set_title("(b) Feature-removal retraining", fontsize=9.2)
    open_axis(axes[1])

    y = np.arange(len(status_values))
    axes[2].barh(
        y,
        status_values,
        color=["#CFE3F5", "#F7E6C4", "#D9D9D9"],
        edgecolor="black",
        hatch=["//", "..", "\\\\"],
        linewidth=0.8,
        height=0.56,
        zorder=3,
    )
    axes[2].set_yticks(y)
    axes[2].set_yticklabels(status_text, fontsize=7.6)
    axes[2].invert_yaxis()
    axes[2].set_xlim(0, max(status_values) + 1)
    axes[2].set_xlabel("Evidence items")
    axes[2].set_title("(c) Evidence closure", fontsize=9.2)
    for yi, val in zip(y, status_values):
        axes[2].text(val + 0.08, yi, str(val), va="center", fontsize=7.5)
    open_axis(axes[2])

    fig.savefig(OUT / "figure8_evidence_closure.svg", bbox_inches="tight")
    fig.savefig(OUT / "figure8_evidence_closure.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
