from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


FIG_DIR = Path("results/cmf_figures")
TABLE_DIR = Path("results/cmf_tables")


def _save(img: Image.Image, stem: str) -> None:
    img.save(FIG_DIR / f"{stem}.png")
    img.convert("P", palette=Image.Palette.ADAPTIVE).save(FIG_DIR / f"{stem}.pdf")


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    names = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for name in names:
        path = Path(name)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _box(draw: ImageDraw.ImageDraw, xy: tuple[int, int, int, int], text: str, fill: tuple[int, int, int], outline=(42, 55, 70)) -> None:
    draw.rounded_rectangle(xy, radius=8, fill=fill, outline=outline, width=2)
    x0, y0, x1, y1 = xy
    lines = text.split("\n")
    font = _font(19, bold=True)
    small = _font(15)
    total_h = 24 + 20 * (len(lines) - 1)
    y = y0 + (y1 - y0 - total_h) // 2
    for i, line in enumerate(lines):
        f = font if i == 0 else small
        bbox = draw.textbbox((0, 0), line, font=f)
        draw.text((x0 + (x1 - x0 - (bbox[2] - bbox[0])) // 2, y), line, fill=(20, 30, 40), font=f)
        y += 24 if i == 0 else 20


def _arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], color=(38, 70, 100)) -> None:
    draw.line((start, end), fill=color, width=3)
    x0, y0 = start
    x1, y1 = end
    if x1 >= x0:
        head = [(x1, y1), (x1 - 12, y1 - 7), (x1 - 12, y1 + 7)]
    else:
        head = [(x1, y1), (x1 + 12, y1 - 7), (x1 + 12, y1 + 7)]
    draw.polygon(head, fill=color)


def architecture_diagram() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (1600, 900), "white")
    draw = ImageDraw.Draw(img)
    title = _font(34, bold=True)
    draw.text((70, 35), "CMF-CAN Architecture", fill=(18, 28, 40), font=title)
    draw.text((70, 82), "Frame / Window / CAN-ID context fusion with anomaly-aware deployment scoring", fill=(70, 82, 96), font=_font(18))

    _box(draw, (70, 180, 280, 290), "CAN Frames\nID, payload, timing", (225, 239, 255))
    _box(draw, (70, 380, 280, 490), "Window Stats\nentropy, rarity, deltas", (226, 246, 235))
    _box(draw, (70, 580, 280, 690), "ID Context\nperiodicity, transitions", (255, 238, 220))

    _box(draw, (410, 160, 690, 310), "Frame Encoder\nID/payload embeddings + Transformer", (225, 239, 255))
    _box(draw, (410, 360, 690, 510), "Stats Encoder\nMLP over window statistics", (226, 246, 235))
    _box(draw, (410, 560, 690, 710), "Context Encoder\nper-frame ID context pooling", (255, 238, 220))

    _box(draw, (820, 250, 1090, 430), "Cross-Modality Fusion\nmodality tokens + cross-attention", (238, 232, 255))
    _box(draw, (820, 510, 1090, 650), "Gated Fusion\nadaptive modality weights", (238, 232, 255))
    _box(draw, (1210, 330, 1450, 470), "CMF Score\nattack probability", (236, 242, 248))
    _box(draw, (1210, 560, 1450, 720), "Anomaly Branch\nPCA, Ledoit-Wolf, tail count,\ntemporal smoothing", (255, 230, 230))

    for y in (235, 435, 635):
        _arrow(draw, (280, y), (410, y))
    _arrow(draw, (690, 235), (820, 310))
    _arrow(draw, (690, 435), (820, 340))
    _arrow(draw, (690, 635), (820, 370))
    _arrow(draw, (955, 430), (955, 510))
    _arrow(draw, (1090, 580), (1210, 630))
    _arrow(draw, (1090, 340), (1210, 400))
    _arrow(draw, (1330, 470), (1330, 560))

    draw.text((1110, 785), "Scenario-aware deployment policy", fill=(55, 65, 78), font=_font(18))
    _save(img, "fig_model_architecture_cmf_can")


def shifted_unknown_chart() -> None:
    df = pd.read_csv(TABLE_DIR / "anomaly_ensemble_final_shifted_summary.csv")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (1500, 820), "white")
    draw = ImageDraw.Draw(img)
    draw.text((70, 35), "Shifted / Unknown Attack Repair", fill=(18, 28, 40), font=_font(32, bold=True))
    draw.text((70, 78), "Baseline CMF-CAN vs final CMF-CAN+Anomaly policy, mean over 3 seeds", fill=(70, 82, 96), font=_font(18))
    metrics = [("best_f1", "F1"), ("best_f1_aupr", "AUPR"), ("best_lowfpr_recall_at_fpr_1em04", "Recall@FPR<=1e-4")]
    left, top, panel_w, panel_h = 90, 150, 410, 560
    colors = {"base": (180, 190, 200), "enh": (52, 116, 202)}
    for pi, (prefix, label) in enumerate(metrics):
        xoff = left + pi * (panel_w + 35)
        draw.text((xoff, top - 42), label, fill=(25, 35, 45), font=_font(22, bold=True))
        y0, y1 = top, top + panel_h
        draw.line((xoff, y1, xoff + panel_w, y1), fill=(90, 95, 105), width=2)
        draw.line((xoff, y0, xoff, y1), fill=(90, 95, 105), width=2)
        for i in range(6):
            y = y1 - panel_h * i / 5
            draw.line((xoff, y, xoff + panel_w, y), fill=(232, 235, 238), width=1)
            draw.text((xoff - 50, y - 8), f"{i/5:.1f}", fill=(80, 85, 90), font=_font(13))
        groups = list(df["dataset"])
        bw, gap = 42, 52
        for gi, dataset in enumerate(groups):
            rec = df[df["dataset"] == dataset].iloc[0]
            base_val = float(rec[f"baseline_{prefix}_mean"]) if f"baseline_{prefix}_mean" in rec else None
            if prefix == "best_f1":
                base_val = float(rec["baseline_f1_mean"])
                enh_val = float(rec["best_f1_mean"])
            elif prefix == "best_f1_aupr":
                base_val = float(rec["baseline_aupr_mean"])
                enh_val = float(rec["best_f1_aupr_mean"])
            else:
                base_val = float(rec["baseline_recall_at_fpr_1em04_mean"])
                enh_val = float(rec["best_lowfpr_recall_at_fpr_1em04_mean"])
            gx = xoff + 38 + gi * (2 * bw + gap)
            for bi, (val, color) in enumerate([(base_val, colors["base"]), (enh_val, colors["enh"])]):
                x = gx + bi * bw
                bar_top = y1 - panel_h * min(val, 1.0)
                draw.rectangle((x, bar_top, x + bw, y1), fill=color)
                draw.text((x - 4, bar_top - 22), f"{val:.2f}", fill=(25, 35, 45), font=_font(13))
            draw.text((gx - 12, y1 + 14), dataset.replace("ctt_", ""), fill=(45, 52, 60), font=_font(14))
    draw.rectangle((1070, 55, 1100, 75), fill=colors["base"])
    draw.text((1110, 52), "Baseline", fill=(50, 60, 70), font=_font(16))
    draw.rectangle((1200, 55, 1230, 75), fill=colors["enh"])
    draw.text((1240, 52), "Enhanced", fill=(50, 60, 70), font=_font(16))
    _save(img, "fig_shifted_unknown_final")


def label_ratio_chart() -> None:
    df = pd.read_csv(TABLE_DIR / "label_ratio_coverage_summary.csv")
    road = df[df["dataset"] == "road"].sort_values("label_ratio")
    img = Image.new("RGB", (1350, 760), "white")
    draw = ImageDraw.Draw(img)
    draw.text((70, 35), "ROAD Few-Label Behavior", fill=(18, 28, 40), font=_font(32, bold=True))
    draw.text((70, 78), "Default F1 winner vs CMF-CAN ranking/low-FPR behavior across label ratios", fill=(70, 82, 96), font=_font(18))
    left, top, right, bottom = 110, 150, 1260, 620
    draw.line((left, bottom, right, bottom), fill=(90, 95, 105), width=2)
    draw.line((left, top, left, bottom), fill=(90, 95, 105), width=2)
    for i in range(6):
        y = bottom - (bottom - top) * i / 5
        draw.line((left, y, right, y), fill=(232, 235, 238), width=1)
        draw.text((55, y - 9), f"{i/5:.1f}", fill=(80, 85, 90), font=_font(14))
    ratios = list(road["label_ratio"])
    bw, gap = 42, 90
    colors = [(52, 116, 202), (42, 151, 99), (225, 130, 43)]
    labels = [("cmf_can_f1_mean", "CMF F1"), ("cmf_can_aupr_mean", "CMF AUPR"), ("cmf_can_recall_at_fpr_1em04_mean", "CMF low-FPR")]
    for idx, rec in enumerate(road.itertuples()):
        gx = left + 55 + idx * (3 * bw + gap)
        for bi, (col, _) in enumerate(labels):
            val = float(getattr(rec, col))
            x = gx + bi * bw
            y = bottom - (bottom - top) * min(val, 1)
            draw.rectangle((x, y, x + bw, bottom), fill=colors[bi])
        draw.text((gx + 15, bottom + 18), f"{rec.label_ratio:g}", fill=(45, 52, 60), font=_font(15))
    for i, (_, label) in enumerate(labels):
        x = 860 + i * 145
        draw.rectangle((x, 55, x + 28, 75), fill=colors[i])
        draw.text((x + 38, 52), label, fill=(50, 60, 70), font=_font(15))
    _save(img, "fig_road_label_ratio_metrics")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    import os

    os.chdir(args.root.resolve())
    architecture_diagram()
    shifted_unknown_chart()
    label_ratio_chart()
    print("[write] paper figures")


if __name__ == "__main__":
    main()
