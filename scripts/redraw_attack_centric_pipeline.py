from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT = Path("results/attack_centric_final/figures")

RED = "#C82127"
RED_DARK = "#7A1115"
RED_LIGHT = "#FFF0EF"
BLUE = "#2F5F8F"
BLUE_LIGHT = "#EAF2FA"
GREEN = "#1B8E69"
GREEN_LIGHT = "#E9F6F0"
AMBER = "#D78300"
AMBER_LIGHT = "#FFF5D9"
INK = "#1F1F1F"
MUTED = "#565656"
GRID = "#CFCFCF"


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "STIX Two Text", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def add_card(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    lines: list[str],
    edge: str,
    fill: str,
    number: str | None = None,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.010,rounding_size=0.018",
        linewidth=1.05,
        edgecolor=edge,
        facecolor=fill,
        zorder=2,
    )
    ax.add_patch(patch)

    title_x = x + 0.032
    if number:
        badge = FancyBboxPatch(
            (x + 0.022, y + h - 0.072),
            0.052,
            0.050,
            boxstyle="round,pad=0.004,rounding_size=0.018",
            linewidth=0.8,
            edgecolor=edge,
            facecolor="white",
            zorder=3,
        )
        ax.add_patch(badge)
        ax.text(x + 0.048, y + h - 0.047, number, ha="center", va="center", fontsize=8.0, fontweight="bold", color=edge)
        title_x = x + 0.086

    ax.text(
        title_x,
        y + h - 0.034,
        title,
        ha="left",
        va="top",
        fontsize=7.85,
        fontweight="bold",
        color=edge,
        zorder=3,
    )

    line_y = y + h - 0.087
    for line in lines:
        ax.text(
            x + 0.036,
            line_y,
            line,
            ha="left",
            va="top",
            fontsize=6.75,
            color=INK,
            zorder=3,
        )
        line_y -= 0.035


def arrow(ax, start: tuple[float, float], end: tuple[float, float], color: str, rad: float = 0.0) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10.0,
            linewidth=1.05,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
            zorder=4,
        )
    )


def pill(ax, x: float, y: float, text: str, edge: str, fill: str = "white", width: float = 0.115) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            0.045,
            boxstyle="round,pad=0.005,rounding_size=0.022",
            linewidth=0.75,
            edgecolor=edge,
            facecolor=fill,
            zorder=3,
        )
    )
    ax.text(x + width / 2, y + 0.0225, text, ha="center", va="center", fontsize=6.45, color=edge, fontweight="bold")


def add_lane_label(ax, x: float, y: float, text: str, color: str) -> None:
    ax.text(x, y, text, ha="center", va="center", fontsize=7.8, color=color, fontweight="bold")
    ax.plot([x - 0.115, x + 0.115], [y - 0.025, y - 0.025], color=color, linewidth=1.0)


def draw_pipeline() -> None:
    setup_style()
    fig, ax = plt.subplots(figsize=(5.05, 3.65))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.035,
        0.965,
        "ACE-CAN attack-centric CAN IDS pipeline",
        ha="left",
        va="top",
        fontsize=10.8,
        fontweight="bold",
        color=INK,
    )
    ax.text(
        0.035,
        0.918,
        "The benchmark first repairs the objective, then tests feature-preserving detection.",
        ha="left",
        va="top",
        fontsize=7.25,
        color=MUTED,
    )

    add_lane_label(ax, 0.160, 0.830, "Input conditions", BLUE)
    add_lane_label(ax, 0.500, 0.830, "Branch operation", MUTED)
    add_lane_label(ax, 0.840, 0.830, "Evidence outputs", MUTED)

    add_card(
        ax,
        0.030,
        0.600,
        0.260,
        0.180,
        "Rare traces",
        ["CAN ID, DLC, payload", "timestamp, file source", "attack rate < 1%"],
        BLUE,
        BLUE_LIGHT,
        "I1",
    )
    add_card(
        ax,
        0.370,
        0.600,
        0.260,
        0.180,
        "Metric audit",
        ["all-normal sanity check", "weighted-score ambiguity", "ranking inversion"],
        RED,
        RED_LIGHT,
        "E1",
    )
    add_card(
        ax,
        0.710,
        0.600,
        0.260,
        0.180,
        "Corrected eval",
        ["attack P/R/F1", "AUPR, Recall@FPR", "event recall"],
        RED,
        RED_LIGHT,
        "E2",
    )

    add_card(
        ax,
        0.030,
        0.315,
        0.260,
        0.180,
        "Cross-shift",
        ["unknown vehicle", "unknown attack family", "normal traffic dominates"],
        BLUE,
        BLUE_LIGHT,
        "I2",
    )
    add_card(
        ax,
        0.370,
        0.315,
        0.260,
        0.180,
        "GRAIN-CAN",
        ["same-ID timing", "payload dynamics", "ID behavior"],
        GREEN,
        GREEN_LIGHT,
        "D1",
    )
    add_card(
        ax,
        0.710,
        0.315,
        0.260,
        0.180,
        "Claim boundary",
        ["strong corrected baseline", "low-FPR evidence", "not solved"],
        GREEN,
        GREEN_LIGHT,
        "D2",
    )

    arrow(ax, (0.290, 0.690), (0.370, 0.690), RED)
    arrow(ax, (0.630, 0.690), (0.710, 0.690), RED)
    arrow(ax, (0.290, 0.405), (0.370, 0.405), GREEN)
    arrow(ax, (0.630, 0.405), (0.710, 0.405), GREEN)

    trap = FancyBboxPatch(
        (0.095, 0.180),
        0.205,
        0.094,
        boxstyle="round,pad=0.008,rounding_size=0.018",
        linewidth=0.85,
        edgecolor=AMBER,
        facecolor=AMBER_LIGHT,
        zorder=3,
    )
    ax.add_patch(trap)
    ax.text(0.1975, 0.242, "metric trap", ha="center", va="center", fontsize=7.2, fontweight="bold", color=AMBER)
    ax.text(0.1975, 0.207, "high aggregate score", ha="center", va="center", fontsize=5.65, color=INK)
    ax.text(0.1975, 0.181, "zero attack recall", ha="center", va="center", fontsize=5.65, color=INK)

    pill(ax, 0.520, 0.216, "Attack-F1", RED, width=0.105)
    pill(ax, 0.635, 0.216, "AUPR", RED, width=0.072)
    pill(ax, 0.717, 0.216, "R@FPR", RED, width=0.083)
    pill(ax, 0.810, 0.216, "Event", RED, width=0.072)

    ax.plot([0.040, 0.960], [0.145, 0.145], color=GRID, linewidth=0.8)
    ax.text(0.500, 0.108, "Output: corrected benchmark + deployment evidence", ha="center", va="center", fontsize=6.65, color=INK, fontweight="bold")
    ax.text(0.500, 0.073, "plus a reproducible reporting checklist", ha="center", va="center", fontsize=6.65, color=INK, fontweight="bold")

    fig.subplots_adjust(left=0.010, right=0.990, bottom=0.020, top=0.985)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "paper_fig9_attack_centric_pipeline.svg", format="svg")
    fig.savefig(OUT / "paper_fig9_attack_centric_pipeline.pdf", format="pdf")
    plt.close(fig)


if __name__ == "__main__":
    draw_pipeline()
