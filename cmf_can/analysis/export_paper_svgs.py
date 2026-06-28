from __future__ import annotations

import argparse
from html import escape
from pathlib import Path

import pandas as pd


TABLE_DIR = Path("results/cmf_tables")
FIG_DIR = Path("results/cmf_figures")


BLUE = "#3b73c6"
GREEN = "#2f9a61"
ORANGE = "#d9812b"
GRAY = "#aeb8c2"
DARK = "#17212f"
GRID = "#dfe4ea"


class SVG:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.items: list[str] = []

    def add(self, text: str) -> None:
        self.items.append(text)

    def text(self, x: float, y: float, text: str, size: int = 14, weight: int = 400, anchor: str = "start", color: str = DARK) -> None:
        self.add(
            f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, Helvetica, sans-serif" '
            f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="{color}">{escape(text)}</text>'
        )

    def line(self, x1: float, y1: float, x2: float, y2: float, color: str = DARK, width: float = 1.4, dash: str | None = None) -> None:
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{width}"{dash_attr}/>')

    def rect(self, x: float, y: float, w: float, h: float, fill: str, stroke: str = "none", width: float = 1.0, rx: float = 0.0, cls: str = "") -> None:
        cls_attr = f' class="{cls}"' if cls else ""
        self.add(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx:.1f}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{width}"{cls_attr}/>'
        )

    def polyline(self, points: list[tuple[float, float]], color: str, width: float = 2.5) -> None:
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        self.add(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round"/>')

    def circle(self, x: float, y: float, r: float, fill: str, stroke: str = "white", width: float = 1.5) -> None:
        self.add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>')

    def arrow(self, x1: float, y1: float, x2: float, y2: float, color: str = "#26384f") -> None:
        self.line(x1, y1, x2, y2, color=color, width=1.8)
        if x2 >= x1:
            pts = [(x2, y2), (x2 - 9, y2 - 5), (x2 - 9, y2 + 5)]
        else:
            pts = [(x2, y2), (x2 + 9, y2 - 5), (x2 + 9, y2 + 5)]
        self.add('<polygon points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y in pts) + f'" fill="{color}"/>')

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        defs = """
<defs>
  <pattern id="hatch" patternUnits="userSpaceOnUse" width="8" height="8" patternTransform="rotate(45)">
    <line x1="0" y1="0" x2="0" y2="8" stroke="#7d8894" stroke-width="1.1"/>
  </pattern>
  <style>
    .axis { stroke: #1f2937; stroke-width: 1.4; }
    .grid { stroke: #dfe4ea; stroke-width: 1.0; }
    .label { font-family: Arial, Helvetica, sans-serif; fill: #334155; }
  </style>
</defs>
"""
        body = "\n".join(self.items)
        path.write_text(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}" viewBox="0 0 {self.width} {self.height}">\n'
            f'{defs}\n<rect width="100%" height="100%" fill="white"/>\n{body}\n</svg>\n',
            encoding="utf-8",
        )


def _axis(svg: SVG, x: float, y: float, w: float, h: float, title: str) -> None:
    svg.text(x, y - 28, title, size=18, weight=700)
    for i in range(6):
        yy = y + h - h * i / 5
        svg.line(x, yy, x + w, yy, color=GRID, width=1)
        svg.text(x - 12, yy + 4, f"{i / 5:.1f}", size=10, anchor="end", color="#53606d")
    svg.line(x, y, x, y + h, color="#111827", width=1.3)
    svg.line(x, y + h, x + w, y + h, color="#111827", width=1.3)


def shifted_unknown_svg() -> None:
    df = pd.read_csv(TABLE_DIR / "anomaly_ensemble_final_shifted_summary.csv")
    svg = SVG(1320, 620)
    svg.text(56, 52, "Shifted / Unknown Attack Repair", size=28, weight=700)
    svg.text(56, 82, "Baseline CMF-CAN vs final CMF-CAN+Anomaly, mean over 3 seeds", size=14, color="#475569")
    svg.rect(970, 38, 22, 14, GRAY)
    svg.rect(970, 38, 22, 14, "url(#hatch)", stroke=GRAY)
    svg.text(1002, 51, "Baseline", size=13, color="#334155")
    svg.rect(1090, 38, 22, 14, BLUE)
    svg.text(1122, 51, "Enhanced", size=13, color="#334155")

    panels = [
        ("F1", "baseline_f1_mean", "best_f1_mean"),
        ("AUPR", "baseline_aupr_mean", "best_f1_aupr_mean"),
        ("Recall@FPR<=1e-4", "baseline_recall_at_fpr_1em04_mean", "best_lowfpr_recall_at_fpr_1em04_mean"),
    ]
    left, top, w, h = 72, 145, 340, 360
    for pi, (title, base_col, enh_col) in enumerate(panels):
        x = left + pi * 410
        _axis(svg, x, top, w, h, title)
        for gi, rec in enumerate(df.itertuples()):
            base = float(getattr(rec, base_col))
            enh = float(getattr(rec, enh_col))
            gx = x + 42 + gi * 95
            bw = 25
            for bi, (val, fill) in enumerate([(base, GRAY), (enh, BLUE)]):
                yy = top + h - h * min(val, 1.0)
                svg.rect(gx + bi * (bw + 6), yy, bw, h * min(val, 1.0), fill)
                if fill == GRAY:
                    svg.rect(gx + bi * (bw + 6), yy, bw, h * min(val, 1.0), "url(#hatch)", stroke=GRAY)
                svg.text(gx + bi * (bw + 6) + bw / 2, yy - 7, f"{val:.2f}", size=10, anchor="middle")
            svg.text(gx + bw + 3, top + h + 24, str(rec.dataset).replace("ctt_", ""), size=11, anchor="middle", color="#334155")
    svg.save(FIG_DIR / "fig_shifted_unknown_final.svg")


def road_label_ratio_svg() -> None:
    df = pd.read_csv(TABLE_DIR / "label_ratio_coverage_summary.csv")
    road = df[df["dataset"] == "road"].sort_values("label_ratio")
    svg = SVG(1120, 620)
    svg.text(56, 52, "ROAD Few-Label Behavior", size=28, weight=700)
    svg.text(56, 82, "CMF-CAN metrics across label ratios; default-F1 winners are reported separately in tables", size=14, color="#475569")
    x, y, w, h = 96, 145, 900, 360
    _axis(svg, x, y, w, h, "Metric value")
    ratios = list(road["label_ratio"])
    xs = [x + i * (w / (len(ratios) - 1)) for i in range(len(ratios))]
    series = [
        ("CMF F1", "cmf_can_f1_mean", BLUE),
        ("CMF AUPR", "cmf_can_aupr_mean", GREEN),
        ("CMF Recall@FPR<=1e-4", "cmf_can_recall_at_fpr_1em04_mean", ORANGE),
    ]
    for si, (name, col, color) in enumerate(series):
        pts = [(xs[i], y + h - h * float(v)) for i, v in enumerate(road[col])]
        svg.polyline(pts, color=color, width=2.8)
        for px, py in pts:
            svg.circle(px, py, 4.5, color)
        lx = 680 + si * 140
        ly = 52
        svg.line(lx, ly - 5, lx + 28, ly - 5, color=color, width=3)
        svg.circle(lx + 14, ly - 5, 4.5, color)
        svg.text(lx + 36, ly, name, size=12, color="#334155")
    for px, ratio in zip(xs, ratios):
        svg.text(px, y + h + 26, f"{ratio:g}", size=12, anchor="middle", color="#334155")
    svg.text(x + w / 2, y + h + 56, "Label ratio", size=13, anchor="middle", color="#334155")
    svg.save(FIG_DIR / "fig_road_label_ratio_metrics.svg")


def architecture_svg() -> None:
    svg = SVG(1380, 780)
    svg.text(56, 52, "CMF-CAN Architecture", size=30, weight=700)
    svg.text(56, 84, "Frame / window / CAN-ID context fusion with anomaly-aware deployment scoring", size=14, color="#475569")
    boxes = [
        (60, 160, 190, 82, "CAN Frames\nID, payload, timing", "#dbeafe"),
        (60, 340, 190, 82, "Window Stats\nentropy, rarity, deltas", "#dcfce7"),
        (60, 520, 190, 82, "ID Context\nperiodicity, transitions", "#ffedd5"),
        (365, 140, 260, 120, "Frame Encoder\nembeddings + Transformer", "#dbeafe"),
        (365, 320, 260, 120, "Stats Encoder\nwindow-stat MLP", "#dcfce7"),
        (365, 500, 260, 120, "Context Encoder\nID-context pooling", "#ffedd5"),
        (750, 230, 270, 150, "Cross-Modality Fusion\nmodality tokens + attention", "#ede9fe"),
        (750, 465, 270, 110, "Gated Fusion\nadaptive modality weights", "#ede9fe"),
        (1120, 300, 220, 105, "CMF Score\nattack probability", "#e2e8f0"),
        (1120, 500, 220, 130, "Anomaly Branch\nPCA, Ledoit-Wolf,\ntail count, smoothing", "#fee2e2"),
    ]
    for x, y, w, h, text, fill in boxes:
        svg.rect(x, y, w, h, fill, stroke="#1f2937", width=1.4, rx=8)
        lines = text.split("\n")
        svg.text(x + w / 2, y + h / 2 - 8 * (len(lines) - 1), lines[0], size=15, weight=700, anchor="middle")
        for i, line in enumerate(lines[1:], 1):
            svg.text(x + w / 2, y + h / 2 + 18 * i - 8 * (len(lines) - 1), line, size=12, anchor="middle")
    for y0 in (201, 381, 561):
        svg.arrow(250, y0, 365, y0)
    svg.arrow(625, 200, 750, 285)
    svg.arrow(625, 380, 750, 305)
    svg.arrow(625, 560, 750, 325)
    svg.arrow(885, 380, 885, 465)
    svg.arrow(1020, 300, 1120, 350)
    svg.arrow(1020, 520, 1120, 565)
    svg.arrow(1230, 405, 1230, 500)
    svg.text(1090, 700, "Scenario-aware deployment policy", size=13, color="#475569")
    svg.save(FIG_DIR / "fig_model_architecture_cmf_can.svg")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    import os

    os.chdir(args.root.resolve())
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    shifted_unknown_svg()
    road_label_ratio_svg()
    architecture_svg()
    print("[write] svg paper figures")


if __name__ == "__main__":
    main()
