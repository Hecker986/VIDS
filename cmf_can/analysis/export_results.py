from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


TABLE_DIR = Path("results/cmf_tables")
FIG_DIR = Path("results/cmf_figures")


def _safe_read(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"[skip] missing {path}")
        return None
    return pd.read_csv(path)


def _write_latex(df: pd.DataFrame, path: Path, cols: Iterable[str]) -> None:
    cols = [c for c in cols if c in df.columns]
    out = df.loc[:, cols].copy()
    for col in out.select_dtypes(include="number").columns:
        out[col] = out[col].map(lambda x: f"{x:.4f}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(out.to_latex(index=False, escape=True), encoding="utf-8")
    print(f"[write] {path}")


def _font(size: int) -> ImageFont.ImageFont:
    for name in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        p = Path(name)
        if p.exists():
            return ImageFont.truetype(str(p), size=size)
    return ImageFont.load_default()


def _bar_chart(
    rows: list[tuple[str, float]],
    path: Path,
    title: str,
    y_label: str,
    width: int = 1100,
    height: int = 620,
) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title_font = _font(26)
    label_font = _font(16)
    tick_font = _font(13)
    left, top, right, bottom = 95, 80, width - 40, height - 115
    draw.text((left, 28), title, fill=(20, 30, 40), font=title_font)
    draw.text((22, top + 150), y_label, fill=(50, 60, 70), font=label_font)
    draw.line((left, bottom, right, bottom), fill=(80, 80, 80), width=2)
    draw.line((left, top, left, bottom), fill=(80, 80, 80), width=2)
    max_val = max(v for _, v in rows)
    y_max = max(1.0, max_val * 1.08)
    for i in range(6):
        y = bottom - (bottom - top) * i / 5
        value = y_max * i / 5
        draw.line((left, y, right, y), fill=(230, 232, 235), width=1)
        draw.text((35, y - 8), f"{value:.2f}", fill=(70, 75, 80), font=tick_font)
    gap = 12
    bar_w = max(12, int((right - left - gap * (len(rows) + 1)) / len(rows)))
    colors = [(48, 112, 214), (225, 130, 43), (42, 151, 99), (150, 88, 180)]
    for idx, (label, value) in enumerate(rows):
        x0 = left + gap + idx * (bar_w + gap)
        x1 = x0 + bar_w
        y0 = bottom - (bottom - top) * value / y_max
        draw.rectangle((x0, y0, x1, bottom), fill=colors[idx % len(colors)])
        draw.text((x0 + 2, y0 - 22), f"{value:.3f}", fill=(25, 25, 25), font=tick_font)
        short = label.replace("concat_fusion", "concat").replace("transformer", "trans")
        draw.text((x0, bottom + 10), short[:18], fill=(35, 35, 35), font=tick_font)
    img.save(path)
    print(f"[write] {path}")


def export_road() -> None:
    main = _safe_read(TABLE_DIR / "road_main_20ep.csv")
    if main is not None:
        main = main.drop_duplicates(["dataset", "model", "seed"], keep="last")
        _write_latex(
            main.sort_values("f1", ascending=False),
            TABLE_DIR / "table_road_main.tex",
            ["model", "f1", "macro_f1", "aupr", "auroc", "fpr", "recall_at_fpr_1em04"],
        )
        _bar_chart(
            [(r.model, float(r.aupr)) for r in main.sort_values("aupr", ascending=False).itertuples()],
            FIG_DIR / "fig_road_main_aupr.png",
            "ROAD Main: AUPR by Model",
            "AUPR",
        )

    few = _safe_read(TABLE_DIR / "road_few_label_3seed_mean_std.csv")
    if few is not None:
        _write_latex(
            few.sort_values(["label_ratio", "model"]),
            TABLE_DIR / "table_road_few_label.tex",
            ["label_ratio", "model", "f1_mean", "f1_std", "aupr_mean", "aupr_std", "recall_at_fpr_1em04_mean"],
        )
        subset = few[few["label_ratio"].isin([0.01, 0.05, 0.10])]
        rows = [(f"{r.model}@{r.label_ratio:g}", float(r.f1_mean)) for r in subset.itertuples()]
        _bar_chart(rows, FIG_DIR / "fig_road_few_label_f1.png", "ROAD Few-Label: F1 Mean", "F1")

    abl = _safe_read(TABLE_DIR / "road_ablation_20ep_merged.csv")
    if abl is not None:
        abl = abl.drop_duplicates(["dataset", "model", "seed"], keep="last")
        _write_latex(
            abl.sort_values("aupr", ascending=False),
            TABLE_DIR / "table_road_ablation.tex",
            ["model", "f1", "aupr", "auroc", "recall_at_fpr_1em04"],
        )

    eff = _safe_read(TABLE_DIR / "efficiency_road.csv")
    if eff is not None:
        _write_latex(
            eff.sort_values("throughput_windows_per_s", ascending=False),
            TABLE_DIR / "table_efficiency_road.tex",
            ["model", "total_params", "avg_batch_ms", "throughput_windows_per_s"],
        )


def export_ctt() -> None:
    gen = _safe_read(TABLE_DIR / "ctt_generalization_15ep.csv")
    if gen is not None:
        gen = gen.drop_duplicates(["dataset", "model", "seed"], keep="last")
        _write_latex(
            gen.sort_values(["dataset", "model"]),
            TABLE_DIR / "table_ctt_generalization.tex",
            ["dataset", "model", "f1", "macro_f1", "aupr", "auroc", "fpr", "recall_at_fpr_1em04"],
        )
    low = _safe_read(TABLE_DIR / "ctt_deployment_low_fpr_15ep.csv")
    if low is not None:
        _write_latex(
            low.sort_values(["dataset", "model"]),
            TABLE_DIR / "table_ctt_deployment_low_fpr.tex",
            ["dataset", "model", "aupr", "auroc", "recall_at_fpr_1em04", "f1_at_fpr_1em04"],
        )
        rows = []
        for r in low[low["dataset"].isin(["ctt_test01", "ctt_test02"])].itertuples():
            if r.model in {"transformer", "concat_fusion", "cmf_can"}:
                rows.append((f"{r.dataset}:{r.model}", float(r.recall_at_fpr_1em04)))
        _bar_chart(rows, FIG_DIR / "fig_ctt_recall_at_fpr_1e4.png", "CT&T Recall at FPR <= 1e-4", "Recall")

    few = _safe_read(TABLE_DIR / "ctt_few_label_3seed_mean_std.csv")
    if few is not None:
        _write_latex(
            few.sort_values(["label_ratio", "model"]),
            TABLE_DIR / "table_ctt_few_label.tex",
            [
                "label_ratio",
                "model",
                "f1_mean",
                "f1_std",
                "aupr_mean",
                "aupr_std",
                "recall_at_fpr_1em04_mean",
                "recall_at_fpr_1em04_std",
            ],
        )
        subset = few[few["label_ratio"].isin([0.01, 0.05, 0.10])]
        rows = [(f"{r.model}@{r.label_ratio:g}", float(r.f1_mean)) for r in subset.itertuples()]
        _bar_chart(rows, FIG_DIR / "fig_ctt_few_label_f1.png", "CT&T Few-Label: F1 Mean", "F1")

    abl = _safe_read(TABLE_DIR / "ctt_ablation_15ep_merged.csv")
    if abl is not None:
        _write_latex(
            abl.sort_values("f1", ascending=False),
            TABLE_DIR / "table_ctt_ablation.tex",
            ["model", "f1", "aupr", "auroc", "recall_at_fpr_1em04", "f1_at_fpr_1em04"],
        )
        rows = [(r.model, float(r.f1)) for r in abl.sort_values("f1", ascending=False).itertuples()]
        _bar_chart(rows, FIG_DIR / "fig_ctt_ablation_f1.png", "CT&T test01 Ablation: F1", "F1")


def export_auxiliary() -> None:
    for stem in ("hcrl_main_15ep", "car_hacking_main_15ep", "crysys_subset_main_15ep"):
        df = _safe_read(TABLE_DIR / f"{stem}.csv")
        if df is not None:
            _write_latex(
                df.drop_duplicates(["dataset", "model", "seed"], keep="last").sort_values("model"),
                TABLE_DIR / f"table_{stem}.tex",
                ["dataset", "model", "f1", "aupr", "auroc", "fpr", "recall_at_fpr_1em04"],
            )
    cry = _safe_read(TABLE_DIR / "crysys_family_mod_3model_3seed_summary.csv")
    if cry is not None:
        _write_latex(
            cry.sort_values("f1_mean", ascending=False),
            TABLE_DIR / "table_crysys_family_mod_3seed.tex",
            [
                "model",
                "n",
                "f1_mean",
                "f1_std",
                "aupr_mean",
                "aupr_std",
                "auroc_mean",
                "auroc_std",
                "recall_at_fpr_1em04_mean",
                "recall_at_fpr_1em04_std",
            ],
        )
        rows = [(r.model, float(r.recall_at_fpr_1em04_mean)) for r in cry.itertuples()]
        _bar_chart(rows, FIG_DIR / "fig_crysys_family_mod_recall_at_fpr_1e4.png", "CrySyS Mod-Only: Recall at FPR <= 1e-4", "Recall")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()
    import os

    os.chdir(root)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    export_road()
    export_ctt()
    export_auxiliary()


if __name__ == "__main__":
    main()
