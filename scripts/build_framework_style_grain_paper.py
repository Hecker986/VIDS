from __future__ import annotations

import os
import shutil
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-vids")

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd


OUT = Path("results/paper_rewrite_framework_style")
FIGS = OUT / "figures"
TABLES = OUT / "tables"
LOGS = OUT / "logs"


BLUE = "#2F5E8E"
BLUE_SOFT = "#B7C9DF"
TEAL = "#2F8C7D"
TEAL_SOFT = "#BFDCD6"
ROSE = "#B96B6C"
ROSE_SOFT = "#E8C8C8"
AMBER = "#B98A2E"
AMBER_SOFT = "#E9D4A8"
GRAY = "#6F7680"
GRAY_SOFT = "#D8DDE3"
LIGHT = "#F6F8FA"
DARK = "#1F2933"
METHOD_COLORS = {
    "GB-sample": "#6F86A5",
    "MLP": AMBER,
    "Raw-Trans": ROSE,
    "GRAIN-W100": TEAL,
}
METHOD_HATCHES = {
    "GB-sample": "",
    "MLP": "///",
    "Raw-Trans": "\\\\",
    "GRAIN-W100": "xx",
}


def setup() -> None:
    for p in [OUT, FIGS, TABLES, LOGS]:
        p.mkdir(parents=True, exist_ok=True)
    for name in ["llncs.cls", "splncs04.bst"]:
        candidates = [
            Path("/home/lqa/.codex/attachments/dc6f4859-68a9-4cca-a8d7-7ebbe7f068b6") / name,
            Path("results/paper_revision_final_grain_can") / name,
            Path("results/final_submission_closure/template") / name,
        ]
        for src in candidates:
            if src.exists():
                shutil.copyfile(src, OUT / name)
                break
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans", "sans-serif"],
            "font.size": 8.2,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def savefig(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIGS / f"{name}.svg", bbox_inches="tight", pad_inches=0.006)
    fig.savefig(FIGS / f"{name}.pdf", bbox_inches="tight", pad_inches=0.006)
    plt.close(fig)


def short_setting(s: str) -> str:
    return {
        "ctt_test01": "Test01",
        "ctt_test02": "Test02",
        "ctt_test03": "Test03",
        "ctt_test04": "Test04",
    }.get(str(s), str(s))


def short_model(m: str, granularity: str | None = None) -> str:
    text = str(m)
    if "all-normal" in text.lower():
        return "All-normal"
    if "MLP" in text:
        return "MLP"
    if "public-style" in text or ("GradientBoosting" in text and str(granularity) == "sample"):
        return "GB-sample"
    if "SAFE_CAN" in text:
        return "SAFE-GB"
    if "window_100" in text or "GRAIN-W100" in text:
        return "GRAIN-W100"
    if "window_10" in text:
        return "GRAIN-W10"
    if "window_20" in text:
        return "GRAIN-W20"
    if "Transformer" in text or "old_window100" in text:
        return "Raw-Trans"
    return text.replace("GradientBoosting / ", "GB-")


def fmt(v) -> str:
    try:
        if pd.isna(v):
            return "NA"
        return f"{float(v):.3f}"
    except Exception:
        return str(v)


def write_csv_tex(df: pd.DataFrame, name: str, caption: str, label: str, cols: list[str] | None = None) -> None:
    df.to_csv(TABLES / f"{name}.csv", index=False)
    use = df.copy() if cols is None else df[cols].copy()
    for c in use.columns:
        if use[c].dtype.kind in "fc":
            use[c] = use[c].map(fmt)
    tex = "\\begin{table}[H]\n\\caption{" + caption + "}\\label{" + label + "}\n\\centering\\small\n"
    tex += use.to_latex(index=False, escape=True).replace("\\toprule", "\\hline").replace("\\midrule", "\\hline").replace("\\bottomrule", "\\hline")
    tex += "\\end{table}\n"
    (TABLES / f"{name}.tex").write_text(tex, encoding="utf-8")


def load_sources():
    conv = pd.read_csv("results/paper_revision_final_grain_can/tables/conventional_ids_metrics_ctt.csv")
    raw = pd.read_csv("results/paper_revision_final_grain_can/tables/raw_window_vs_grain.csv")
    abl = pd.read_csv("results/paper_revision_final_grain_can/tables/grain_feature_ablation.csv")
    win = pd.read_csv("results/paper_revision_final_grain_can/tables/window_length_sensitivity.csv")
    supp = pd.read_csv("results/paper_revision_final_grain_can/tables/supplementary_operating_metrics_ctt.csv")
    data = pd.read_csv("results/paper_revision_final_grain_can/tables/dataset_summary.csv")
    return conv, raw, abl, win, supp, data


def build_tables():
    conv, raw, abl, win, supp, data = load_sources()

    pipeline = pd.DataFrame(
        [
            ["Public GB", "sample public-style features", "GB", "classical reproduced baseline"],
            ["Public MLP", "sample public-style features", "MLP", "neural classical baseline"],
            ["Raw-Trans", "raw fixed frame sequence, W100", "Transformer", "raw-window deep baseline"],
            ["GRAIN-W100", "same-ID local behavior features, W100", "GB", "proposed detector pipeline"],
        ],
        columns=["Detector", "Representation", "Classifier", "Purpose"],
    )
    write_csv_tex(pipeline, "table1_compared_pipelines", "Compared IDS pipelines used in the main evaluation.", "tab:pipelines")

    role = {
        "ctt_test01": ("known", "known", "known-vehicle known-attack baseline"),
        "ctt_test02": ("unknown", "known", "vehicle-shift stress test"),
        "ctt_test03": ("known", "unknown", "attack-shift stress test"),
        "ctt_test04": ("unknown", "unknown", "joint vehicle/attack-shift stress test"),
    }
    ds_rows = []
    for s, (veh, atk, r) in role.items():
        d = data[data["Dataset"].eq(s)].head(1)
        if len(d):
            ds_rows.append(
                {
                    "Setting": short_setting(s),
                    "Vehicle shift": veh,
                    "Attack shift": atk,
                    "Frames": int(d.iloc[0]["Frames"]),
                    "Positive rate": float(d.iloc[0]["Attack ratio"]),
                    "Role": r,
                }
            )
    ds = pd.DataFrame(ds_rows)
    write_csv_tex(ds, "table2_dataset_shift_settings", "CT\\&T cross-shift settings.", "tab:settings")

    keep = []
    for _, r in conv.iterrows():
        sm = short_model(r["model"], r.get("granularity"))
        if sm not in {"All-normal", "GB-sample", "MLP", "Raw-Trans", "SAFE-GB", "GRAIN-W100"}:
            continue
        if sm == "GRAIN-W100" and str(r.get("granularity")) != "window_100":
            continue
        if sm == "GB-sample" and str(r.get("source")) != "final_grain_can_a1_negative_stability":
            continue
        keep.append(
            {
                "Setting": short_setting(r["setting"]),
                "Detector": sm,
                "P": r["precision"],
                "R": r["recall"],
                "F1": r["f1"],
                "FNR": r["fnr"],
                "FPR": r["fpr"],
                "AUPR": r["aupr"],
                "AUROC": r["auroc"],
                "R@1e-3": r["detection_rate_at_fpr_1e_3"],
                "Positive rate": r["positive_rate"],
            }
        )
    main = pd.DataFrame(keep)
    # Keep the table compact: all four settings, four core detectors.
    order = ["GB-sample", "MLP", "Raw-Trans", "GRAIN-W100"]
    main = main[main["Detector"].isin(order)]
    main["ord"] = main["Detector"].map({m: i for i, m in enumerate(order)})
    main = main.sort_values(["Setting", "ord"]).drop_duplicates(["Setting", "Detector"], keep="last").drop(columns=["ord"])
    write_csv_tex(main, "table3_main_ctt_results", "Main CT\\&T attack-positive results.", "tab:main")

    raw_table = raw.copy()
    raw_table["Setting"] = raw_table["setting"].map(short_setting)
    raw_table["Pipeline"] = raw_table["model"].map(lambda x: "GRAIN-W100" if "GRAIN" in str(x) else "Raw-Trans")
    raw_table = raw_table[["Setting", "Pipeline", "precision", "recall", "f1", "aupr"]].rename(
        columns={"precision": "P", "recall": "R", "f1": "F1", "aupr": "AUPR"}
    )
    write_csv_tex(raw_table, "table4_raw_vs_grain", "Raw fixed-window baseline versus GRAIN-CAN.", "tab:rawgrain")

    wanted_abl = {
        "full_safe_can": "Full safe features",
        "only_timing": "Timing only",
        "without_delta_t_same_id": "w/o same-ID timing",
        "without_payload_delta_l1": "w/o payload delta",
        "without_payload_statistics": "w/o payload stats",
        "without_can_id": "w/o CAN ID",
    }
    ab = abl[abl["ablation"].isin(wanted_abl)].copy()
    ab["Variant"] = ab["ablation"].map(wanted_abl)
    ab = ab[["Variant", "precision", "recall", "f1", "fnr", "fpr", "aupr", "detection_rate_at_fpr_1e_3"]].rename(
        columns={"precision": "P", "recall": "R", "f1": "F1", "fnr": "FNR", "fpr": "FPR", "aupr": "AUPR", "detection_rate_at_fpr_1e_3": "R@1e-3"}
    )
    write_csv_tex(ab, "table5_feature_ablation", "Feature-group ablation on CT\\&T Test04.", "tab:ablation")

    comp = pd.DataFrame(
        [
            ["CANShield", "signal-level multi-scale autoencoder ensemble", "DBC / signal extraction", "GRAIN uses raw fields and no AE ensemble"],
            ["X-CANIDS", "signal-aware explainable detection", "CAN database / signal semantics", "GRAIN does not require decoded signals"],
            ["CANet", "message-level unsupervised neural detector", "raw CAN messages", "GRAIN is supervised and window-aggregated"],
            ["MIDS", "bidirectional Mamba sequence model", "message sequence", "GRAIN is a feature pipeline, not a deep backbone"],
            ["GRAIN-CAN", "same-ID local behavior before aggregation", "timestamp, ID, DLC, payload", "proposed lightweight supervised pipeline"],
        ],
        columns=["Work", "Core mechanism", "Required information", "Difference"],
    )
    write_csv_tex(comp, "table6_pipeline_positioning", "Positioning against representative CAN IDS pipeline families.", "tab:positioning")

    return conv, raw, ab, win, supp, main


def add_box(ax, xy, wh, text, fc, ec, fs=7.8, weight="bold"):
    box = patches.FancyBboxPatch(
        xy,
        wh[0],
        wh[1],
        boxstyle="round,pad=0.018,rounding_size=0.02",
        linewidth=1.15,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(box)
    ax.text(xy[0] + wh[0] / 2, xy[1] + wh[1] / 2, text, ha="center", va="center", fontsize=fs, fontweight=weight, color=DARK, linespacing=1.08)


def draw_pipeline():
    # Claim: GRAIN-CAN is a clean detector pipeline, not an evaluation framework.
    fig, ax = plt.subplots(figsize=(7.25, 2.05))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.02, 0.93, "GRAIN-CAN detector pipeline", fontsize=12.4, fontweight="bold", ha="left", color=DARK)
    ax.text(0.02, 0.82, "Same-ID behavior is extracted before fixed-window classification.", fontsize=8.6, color="#56616F", ha="left")
    labels = [
        ("Raw CAN\nstream\n$t$, ID, DLC,\npayload", "#EEF3F8", BLUE),
        ("Per-ID\nhistory\nlast time,\nlast payload", "#EEF3F8", BLUE),
        ("Local\nresiduals\ntiming, payload,\nID behavior", "#EAF5F2", TEAL),
        ("Window\naggregation\nfixed W", "#F8F2E4", AMBER),
        ("Detector\nscore + label", "#F7EEEE", ROSE),
    ]
    x0, y, w, h, gap = 0.028, 0.28, 0.142, 0.42, 0.055
    for i, (txt, fc, ec) in enumerate(labels):
        x = x0 + i * (w + gap)
        add_box(ax, (x, y), (w, h), txt, fc, ec, fs=7.4)
        if i < len(labels) - 1:
            ax.annotate(
                "",
                xy=(x + w + gap - 0.006, y + h / 2),
                xytext=(x + w + 0.006, y + h / 2),
                arrowprops=dict(arrowstyle="-|>", lw=1.65, color="#505A64", mutation_scale=15),
            )
    ax.text(0.50, 0.11, "History-only extraction: no future frames, no test labels, no test-fitted statistics.", ha="center", fontsize=7.8, color="#56616F")
    savefig(fig, "fig1_grain_pipeline")


def draw_mechanism():
    # Claim: local residual extraction makes sparse attack evidence visible before aggregation.
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 2.35), sharey=True)
    for ax in axes:
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[:].set_visible(False)
    n = 56
    x = np.arange(n)
    attack = (x >= 29) & (x <= 33)
    axes[0].bar(x, np.where(attack, 0.20, 0.15), color=np.where(attack, ROSE, GRAY_SOFT), width=0.9, edgecolor="none")
    axes[0].set_title("(a) Raw long window", fontsize=9.8, fontweight="bold", pad=2)
    axes[0].text(0.5, 0.78, "sparse attack frames are diluted", ha="center", transform=axes[0].transAxes, fontsize=9.0, color="#39414A")
    residual = np.random.default_rng(3).normal(0.09, 0.015, n)
    residual[attack] = [0.46, 0.58, 0.74, 0.62, 0.48]
    axes[1].bar(x, residual, color=np.where(attack, ROSE, TEAL), width=0.9, edgecolor="none", alpha=0.90)
    axes[1].set_title("(b) Same-ID residual features", fontsize=9.8, fontweight="bold", pad=2)
    axes[1].text(0.5, 0.78, "local residuals preserve the event", ha="center", transform=axes[1].transAxes, fontsize=9.0, color="#39414A")
    for ax in axes:
        ax.axhline(0.3, color="#A8B0BA", lw=0.8, ls="--")
    fig.tight_layout(w_pad=1.0, pad=0.2)
    savefig(fig, "fig2_local_evidence_mechanism")


def draw_main_results(main: pd.DataFrame):
    # Claim: GRAIN is stable under test02/test04 compared with raw windows.
    models = ["GB-sample", "MLP", "Raw-Trans", "GRAIN-W100"]
    settings = ["Test01", "Test02", "Test03", "Test04"]
    fig, ax = plt.subplots(figsize=(7.25, 2.75))
    x = np.arange(len(settings))
    width = 0.18
    offsets = (np.arange(len(models)) - (len(models) - 1) / 2) * width
    for off, m in zip(offsets, models):
        vals = []
        for s in settings:
            d = main[(main["Setting"].eq(s)) & (main["Detector"].eq(m))]
            vals.append(float(d["F1"].iloc[0]) if len(d) else np.nan)
        bars = ax.bar(x + off, vals, width=width * 0.92, label=m, color=METHOD_COLORS[m], edgecolor="#252B33", linewidth=0.45)
        for b in bars:
            b.set_hatch(METHOD_HATCHES[m])
    ax.set_xticks(np.arange(len(settings)))
    ax.set_xticklabels(settings)
    ax.set_ylabel("Attack F1")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", color="#E8EBEF", lw=0.55)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.10), fontsize=7.6, handlelength=1.5, columnspacing=1.2)
    savefig(fig, "fig3_main_ctt_results")


def draw_raw_vs_grain(raw: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7.25, 2.55))
    d = raw.copy()
    d["Setting"] = d["setting"].map(short_setting)
    settings = ["Test01", "Test02", "Test03", "Test04"]
    series = {}
    for label, selector in [("GRAIN-W100", d["model"].eq("GRAIN-W100")), ("Raw-Trans", d["model"].str.contains("Old raw", na=False))]:
        g = d[selector].set_index("Setting")
        series[label] = [float(g.loc[s, "f1"]) if s in g.index else np.nan for s in settings]
    x = np.arange(len(settings))
    width = 0.31
    for off, label in [(-width / 2, "Raw-Trans"), (width / 2, "GRAIN-W100")]:
        bars = ax.bar(x + off, series[label], width=width, label=label, color=METHOD_COLORS[label], edgecolor="#252B33", linewidth=0.5)
        for b in bars:
            b.set_hatch(METHOD_HATCHES[label])
    ax.set_xticks(x)
    ax.set_xticklabels(settings)
    ax.set_ylabel("Attack F1")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", color="#E8EBEF", lw=0.55)
    ax.legend(loc="upper right", fontsize=7.8, ncol=2)
    savefig(fig, "fig4_raw_vs_grain")


def draw_ablation(ab: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7.25, 2.75))
    order = ["Full safe features", "Timing only", "w/o CAN ID", "w/o payload delta", "w/o payload stats", "w/o same-ID timing"]
    d = ab.set_index("Variant").reindex(order).reset_index()
    y = np.arange(len(d))
    base = mpl.colors.to_rgba(TEAL)
    colors = []
    for v in d["Variant"]:
        if v == "Full safe features":
            colors.append(TEAL)
        elif v == "Timing only":
            colors.append(BLUE)
        else:
            colors.append((base[0], base[1], base[2], 0.30))
    ax.barh(y, d["F1"], color=colors, edgecolor="#252B33", lw=0.45)
    ax.set_yticks(y)
    ax.set_yticklabels(d["Variant"], fontsize=7.8)
    ax.invert_yaxis()
    ax.set_xlabel("Attack F1")
    ax.set_xlim(0, max(0.72, float(d["F1"].max()) * 1.12))
    ax.grid(axis="x", color="#E8EBEF", lw=0.55)
    for i, v in enumerate(d["F1"]):
        ax.text(v + 0.012, i, f"{v:.3f}", va="center", fontsize=7.5, color="#303943")
    savefig(fig, "fig5_feature_ablation")


def draw_low_fpr(raw: pd.DataFrame, supp: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7.25, 2.45))
    corrected_path = Path("results/metric_correction_paper/tables/corrected_test04_benchmark_final.csv")
    corrected = pd.read_csv(corrected_path) if corrected_path.exists() else pd.DataFrame()
    rows = []
    for label, model_filter in [("Raw-Trans", "old_window100"), ("GRAIN-W100", "window_100")]:
        if label == "Raw-Trans":
            d = supp[(supp["setting"].eq("ctt_test04")) & (supp["model"].str.contains("old_window100", na=False))]
            aupr = float(d["aupr"].iloc[0]) if len(d) else np.nan
            r = float(d["detection_rate_at_fpr_1e_3"].iloc[0]) if len(d) else np.nan
        else:
            d = raw[(raw["setting"].eq("ctt_test04")) & (raw["model"].eq("GRAIN-W100"))]
            aupr = float(d["aupr"].iloc[0]) if len(d) else np.nan
            method_col = "method" if "method" in corrected.columns else "model"
            c = corrected[corrected[method_col].astype(str).str.contains("GRAIN_window_100", na=False)] if len(corrected) and method_col in corrected.columns else pd.DataFrame()
            r = float(c["recall_at_fpr_1e_3"].iloc[0]) if len(c) and "recall_at_fpr_1e_3" in c.columns else np.nan
        rows.append({"Model": label, "AUPR": aupr, "R@1e-3": r})
    op = pd.DataFrame(rows)
    op.to_csv(TABLES / "figure6_operating_points.csv", index=False)
    vals = op.set_index("Model")[["AUPR", "R@1e-3"]].loc[["Raw-Trans", "GRAIN-W100"]]
    y = np.arange(len(vals.index))
    height = 0.32
    bars1 = ax.barh(y - height / 2, vals["AUPR"], height=height, color=BLUE_SOFT, edgecolor="#252B33", lw=0.45, label="AUPR")
    bars2 = ax.barh(y + height / 2, vals["R@1e-3"], height=height, color=TEAL, edgecolor="#252B33", lw=0.45, label="R@1e-3")
    for b in bars1:
        b.set_hatch("//")
    for b in bars2:
        b.set_hatch("")
    ax.set_yticks(y)
    ax.set_yticklabels(vals.index)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Metric value")
    ax.grid(axis="x", color="#E8EBEF", lw=0.55)
    for bars in [bars1, bars2]:
        for b in bars:
            ax.text(b.get_width() + 0.018, b.get_y() + b.get_height() / 2, f"{b.get_width():.2f}", va="center", fontsize=7.6, color="#303943")
    ax.legend(ncol=2, loc="lower right", fontsize=7.8)
    savefig(fig, "fig6_low_fpr_operating")


def build_figures(conv, raw, ab, win, supp, main):
    draw_pipeline()
    draw_mechanism()
    draw_main_results(main)
    draw_raw_vs_grain(raw)
    draw_ablation(ab)
    draw_low_fpr(raw, supp)


def write_bib():
    src = Path("results/paper_revision_final_grain_can/grain_can_refs.bib")
    text = src.read_text(encoding="utf-8")
    extra = r"""

@article{xcanids2023,
  author  = {Jeong, Seonghoon and Lee, Sangho and Lee, Hwejae and Kim, Huy Kang},
  title   = {{X-CANIDS}: Signal-Aware Explainable Intrusion Detection System for Controller Area Network-Based In-Vehicle Network},
  journal = {arXiv preprint arXiv:2303.12278},
  year    = {2023}
}

@article{canet2019,
  author  = {Hanselmann, Markus and Strauss, Thilo and Dormann, Katharina and Ulmer, Holger},
  title   = {{CANet}: An Unsupervised Intrusion Detection System for High Dimensional {CAN} Bus Data},
  journal = {arXiv preprint arXiv:1906.02492},
  year    = {2019}
}

@article{blevins2021time,
  author  = {Blevins, Deborah H. and Moriano, Pablo and Bridges, Robert A. and Verma, Miki E. and Iannacone, Michael D. and Hollifield, Samuel C.},
  title   = {Time-Based {CAN} Intrusion Detection Benchmark},
  journal = {arXiv preprint arXiv:2101.05781},
  year    = {2021}
}
"""
    (OUT / "grain_can_framework_refs.bib").write_text(text + extra, encoding="utf-8")


def latex_table_from_csv(name: str, caption: str, label: str, cols: list[str] | None = None, resize: bool = False) -> str:
    df = pd.read_csv(TABLES / f"{name}.csv")
    if cols:
        df = df[cols]
    for c in df.columns:
        if df[c].dtype.kind in "fc":
            df[c] = df[c].map(fmt)
    body = "\\begin{table}[H]\n\\caption{" + caption + "}\\label{" + label + "}\n\\centering\\small\n"
    if name in {"table1_compared_pipelines", "table6_pipeline_positioning"}:
        spec = {
            "table1_compared_pipelines": r">{\raggedright\arraybackslash}p{0.14\textwidth}>{\raggedright\arraybackslash}p{0.31\textwidth}>{\raggedright\arraybackslash}p{0.18\textwidth}>{\raggedright\arraybackslash}p{0.27\textwidth}",
            "table6_pipeline_positioning": r">{\raggedright\arraybackslash}p{0.13\textwidth}>{\raggedright\arraybackslash}p{0.31\textwidth}>{\raggedright\arraybackslash}p{0.22\textwidth}>{\raggedright\arraybackslash}p{0.25\textwidth}",
        }[name]
        body += "\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{" + spec + "}\n\\hline\n"
        body += " & ".join(df.columns) + " \\\\\n\\hline\n"
        for _, row in df.iterrows():
            body += " & ".join(str(row[c]).replace("_", "\\_") for c in df.columns) + " \\\\\n"
        body += "\\hline\n\\end{tabular}\n\\end{table}\n"
        return body
    tab = df.to_latex(index=False, escape=True).replace("\\toprule", "\\hline").replace("\\midrule", "\\hline").replace("\\bottomrule", "\\hline")
    if resize:
        body += "\\resizebox{\\textwidth}{!}{%\n" + tab + "}\n"
    else:
        body += tab
    body += "\\end{table}\n"
    return body


def write_latex():
    tex = r"""\documentclass[runningheads]{llncs}
\usepackage[T1]{fontenc}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{url}
\usepackage{booktabs}
\usepackage{lmodern}
\usepackage{array}
\usepackage{float}
\usepackage[section]{placeins}
\emergencystretch=2em
\newcommand{\method}{GRAIN-CAN}
\begin{document}
\raggedbottom
\title{\method{}: Same-ID Local Behavior Features for Cross-Vehicle CAN Intrusion Detection}
\titlerunning{\method{}}
\author{Qi'ao Li}
\authorrunning{Q. Li}
\institute{School of Cyber Science and Technology, Beihang University, Beijing, China\\
\email{qi\_aolee@buaa.edu.cn}}
\maketitle
\begin{abstract}
Cross-vehicle CAN intrusion detection is difficult because vehicle-specific message distributions and attack patterns can both shift between training and deployment. Raw fixed-window representations further obscure short attack bursts when only a few frames in a long window carry abnormal evidence. This paper proposes \method{}, a supervised feature-based CAN IDS pipeline that exposes same-ID local behavior changes before window aggregation. For each CAN frame, \method{} maintains a per-ID history state and extracts timing residuals, payload changes, payload statistics, and local ID-behavior features using only past traffic. These frame-level features are aggregated with a fixed pre-test window configuration and classified by a lightweight supervised detector. We evaluate \method{} on CT\&T test01--test04 using attack-positive precision, recall, F1, FNR, FPR, and confusion matrices, with AUPR and fixed-FPR detection rates reported as supplementary score-based metrics. The results show that same-ID local behavior features improve over raw fixed-window baselines in the hardest shifted settings, while ablations identify timing residuals as the strongest signal in the current setting.
\keywords{CAN bus \and Intrusion detection \and Automotive security \and Feature pipeline}
\end{abstract}

\section{Introduction}
Modern vehicles depend on CAN buses for communication among electronic control units, yet CAN was not designed with message authentication or origin verification. Practical vehicle-security studies have shown that this design leaves room for malicious message injection and remote attack paths~\cite{koscher2010experimental,checkoway2011comprehensive,miller2015remote}. CAN intrusion detection systems therefore monitor bus traffic and infer abnormal behavior from timestamps, arbitration IDs, DLCs, and payload bytes.

Cross-vehicle detection is harder than closed-split detection. A detector trained on one vehicle may see different ID frequencies, payload ranges, and timing profiles on another vehicle. Unknown-attack settings add a second shift: the attack family and script may also differ. Raw fixed-window detectors face an additional problem. If a short attack burst affects only a few frames inside a long window, the abnormal evidence can be diluted before classification.

\method{} is built around a simple design insight: before constructing a window, convert each frame into local behavior changes relative to recent history of the same CAN ID. The detector keeps a per-ID history state, extracts timing residuals, payload-change residuals, payload statistics, and local ID-behavior features, then aggregates those features under a fixed pre-test window configuration. The supervised classifier is intentionally lightweight; the representation before aggregation is the contribution.

The paper makes two contributions. First, we propose \method{}, a supervised feature-based CAN IDS pipeline that exposes same-ID local behavior before fixed-window aggregation. Second, we provide a controlled CT\&T cross-shift evaluation that compares pipeline-level representations, including public-style classical features, raw fixed-window Transformer results, safe-feature baselines, and GRAIN windows, using conventional IDS metrics and feature ablations.

\section{Background and Design Motivation}
\subsection{CAN Intrusion Detection Under Cross-Shift}
CAN IDS methods use timing, payload evolution, ID frequencies, transition structure, learned sequence models, signal-level representations, or physical-layer fingerprints~\cite{song2016intrusion,taylor2016frequency,cho2016fingerprinting,cho2017viden}. Recent work has explored BERT-style sequence modeling, signal-level autoencoder frameworks, graph-style representations, federated learning, and Mamba-style sequence models~\cite{alkhatib2022canbert,shahriar2022canshield,wang2023statgraph,althunayyan2024federated,liu2026mids}. Benchmark datasets such as ROAD and CT\&T make cross-setting evaluation more concrete~\cite{verma2020road,blevins2021time,lampe2024ctt,guerra2024road}.

\subsection{Why Raw Fixed Windows Lose Local Attack Evidence}
Many CAN attacks are local in both time and ID space. A spoofing or injection burst may alter only a handful of messages, while the surrounding window remains normal. A raw fixed-window sequence model must learn both the local perturbation and how to keep it visible after aggregation or pooling. This is possible, but the representation burden is high when the vehicle distribution changes.

\subsection{Design Insight: Same-ID Local Behavior Changes}
\method{} starts from the observation that CAN IDs often carry repeated behavior. Comparing a frame with the recent history of the same ID exposes residuals: timing gaps, payload continuity changes, payload summary shifts, and local concentration changes. These residuals are not vehicle-independent, but they are less tied to absolute payload or frequency values than raw windows alone.

\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{figures/fig1_grain_pipeline.pdf}
\caption{\method{} is a detector pipeline: it converts raw CAN frames into same-ID local behavior features before fixed-window supervised classification.}
\label{fig:pipeline}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{figures/fig2_local_evidence_mechanism.pdf}
\caption{Feature-before-window extraction keeps sparse local attack evidence visible before aggregation, whereas raw long windows can numerically dilute the same event.}
\label{fig:mechanism}
\end{figure}

\section{\method{} Pipeline}
\subsection{Overview}
\method{} has four stages: per-ID history maintenance, frame-level same-ID local behavior extraction, fixed pre-test window aggregation, and supervised classification. Unlike raw-window neural detectors that learn directly from a fixed sequence of frames, \method{} first converts each frame into local behavior residuals relative to the recent history of the same CAN ID.

\subsection{Input Stream and Per-ID History State}
The input stream is \(x_i=(t_i,id_i,dlc_i,b_i)\), where \(t_i\) is the timestamp, \(id_i\) is the arbitration ID, \(dlc_i\) is the data length, and \(b_i\) denotes the payload bytes. For every ID, \method{} maintains a compact state consisting of the last timestamp, last payload, recent counters, and recent summary statistics. The state is updated online and uses only current and past frames.

\subsection{Same-ID Local Behavior Feature Extraction}
The extracted features are grouped by purpose. Timing residuals capture disruptions in periodicity or burst patterns. Payload-change residuals capture discontinuities in byte-level behavior. Payload statistics summarize coarse byte distribution without requiring DBC decoding. Local ID-behavior features capture short concentration and frequency changes. These groups are later aggregated into fixed windows.

\subsection{Window Construction and Labeling}
Frame features are aggregated into fixed windows such as W10, W20, or W100. The window size is a configuration fixed before testing. A window is labelled attack if any frame in the window is labelled attack. This rule is common for window-level IDS experiments, but it also means rare attack windows can contain many normal frames; the design of \method{} is intended to keep the local evidence visible before the label is applied.

\subsection{Supervised Detector and Score Output}
In the repository experiments, \method{} uses a gradient-boosting tree ensemble. The classifier is not the main contribution; it learns a joint decision boundary over interpretable same-ID local behavior features. When the classifier exposes a score, we report AUPR and fixed-FPR detection rates as supplementary operating evidence.
"""
    tex += latex_table_from_csv("table1_compared_pipelines", "Compared IDS pipelines. GRAIN-CAN differs in representation before aggregation rather than in classifier complexity.", "tab:pipelines", resize=True)
    tex += r"""
\subsection{Training and Inference Protocol}
\paragraph{Algorithm 1: training.}
Given a fixed window size \(W\), initialize an empty per-ID history table. Process the training stream in timestamp order. For each frame, read the current state of its CAN ID, compute local behavior features, update the state with the current frame, and append the feature vector to the current window. After every \(W\) frames, aggregate the frame features into a window vector and assign an attack label if any frame in the window is attack. Fit the supervised detector on training windows and, if a score threshold is required, select it on validation data.

\paragraph{Algorithm 1: inference.}
Freeze the feature definitions, window size, detector, and threshold. Process the test stream once in timestamp order. For each frame, compute features from current and past traffic only, update the per-ID state, aggregate windows, and emit a score or label. Test labels are used only after inference for evaluation.

\subsection{Relation to Existing Pipeline Types}
\method{} is not intended to replace signal-level, graph, self-supervised, or deep sequence IDS frameworks. Its scope is narrower: a lightweight supervised representation pipeline that exposes same-ID local behavior before aggregation and can be evaluated under cross-shift settings.
"""
    tex += latex_table_from_csv("table6_pipeline_positioning", "Positioning against representative CAN IDS pipeline families.", "tab:positioning", resize=True)
    tex += r"""
\section{Related Work}
\subsection{Rule, Statistical, and Timing CAN IDS}
Early CAN IDS work often used message timing, ID frequency, and interval statistics because many injection attacks perturb periodic traffic. These approaches are attractive for deployment because they are lightweight and interpretable~\cite{song2016intrusion,taylor2016frequency,blevins2021time}. Their weakness is that a single statistic can miss attacks that preserve timing or shift behavior across vehicles. GRAIN-CAN keeps the lightweight spirit but uses multiple local residual features jointly through a supervised classifier.

\subsection{Deep Sequence and Representation Learning}
Deep CAN IDS methods learn from raw message sequences, payload bits, language-model-style tokenization, or modern sequence backbones~\cite{alkhatib2022canbert,liu2026mids}. These methods can learn rich dependencies but often require larger training sets, careful protocol design, and more compute. GRAIN-CAN takes a different route: it exposes local behavior changes with explicit features before any classifier sees a window.

\subsection{Signal-Level and Explainable Frameworks}
Signal-level frameworks such as CANShield and X-CANIDS use decoded signals or semantic signal structure to improve detection and interpretability~\cite{shahriar2022canshield,xcanids2023}. This is powerful when DBC or signal-level information is available. GRAIN-CAN uses only raw CAN fields and same-ID history, so it is easier to apply when signal definitions are unavailable.

\subsection{Unsupervised and Historical Models}
Unsupervised and historical models such as CANet learn normal message behavior without attack labels~\cite{canet2019}. GRAIN-CAN is supervised and should not be framed as solving arbitrary zero-day detection. Its value is narrower: it tests whether history-based local residuals improve cross-shift supervised detection when labels exist.

\section{Experimental Setup}
\subsection{Datasets and Cross-Shift Settings}
CT\&T set01 is the main dataset. Test01 uses a known vehicle and known attack families; Test02 changes the vehicle; Test03 changes the attack family; Test04 changes both. This gives a compact vehicle-shift and attack-shift matrix.
"""
    tex += latex_table_from_csv("table2_dataset_shift_settings", "CT\\&T cross-shift settings used in the main evaluation.", "tab:settings", resize=True)
    tex += r"""
\subsection{Compared IDS Pipelines}
The main comparison keeps only representative detectors: public-style classical GradientBoosting and MLP rows, a raw fixed-window Transformer, and GRAIN-CAN windows. Simple single-signal thresholds are treated as diagnostic checks outside the main figure.

\subsection{Metrics and Threshold Rules}
Attack is the positive class. We report precision, recall, F1, FNR, and FPR from the confusion matrix. AUPR, AUROC, and recall at fixed FPR are supplementary because they require comparable scores. Window size and threshold must be fixed before test evaluation.

\subsection{Implementation Details}
The GRAIN rows use history-only features over timestamp, ID, DLC, payload bytes, same-ID timing residuals, payload-change residuals, payload statistics, and local ID behavior. W10, W20, and W100 are reported as configurations rather than test-selected winners.

\section{Evaluation}
\subsection{Cross-Shift Detection Results}
Table~\ref{tab:main} and Fig.~\ref{fig:main} summarize the main cross-shift results. The key pattern is not that one detector wins every setting. Rather, GRAIN-CAN provides strong attack-positive detection under vehicle and attack shifts, especially compared with raw fixed-window behavior in Test02 and Test04.
"""
    tex += latex_table_from_csv("table3_main_ctt_results", "Main CT\\&T results with attack as the positive class.", "tab:main", cols=["Setting", "Detector", "P", "R", "F1", "FNR", "FPR"])
    tex += r"""
\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{figures/fig3_main_ctt_results.pdf}
\caption{GRAIN-CAN remains competitive across CT\&T shifts, while the raw fixed-window Transformer collapses in vehicle-shifted settings.}
\label{fig:main}
\end{figure}

\subsection{Effect of Feature-Before-Window Representation}
The raw-window comparison isolates the representation effect. Both pipelines use fixed windows, but only GRAIN-CAN extracts same-ID local behavior before aggregation. The contrast is strongest in Test02 and Test04, where raw windows produce high miss or false-alarm behavior.
"""
    tex += latex_table_from_csv("table4_raw_vs_grain", "Raw fixed-window pipeline versus GRAIN-CAN.", "tab:rawgrain")
    tex += r"""
\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{figures/fig4_raw_vs_grain.pdf}
\caption{The same W100 aggregation becomes more reliable when local behavior residuals are computed before the window is formed.}
\label{fig:rawgrain}
\end{figure}

\subsection{Contribution of Local Behavior Features}
Feature-group ablation on Test04 shows that same-ID timing is the strongest individual signal in the current setting. Removing same-ID timing sharply degrades F1, while payload and ID groups have weaker effects. This supports the pipeline's focus on residual behavior rather than absolute payload values alone.
"""
    tex += latex_table_from_csv("table5_feature_ablation", "Feature-group ablation on CT\\&T Test04.", "tab:ablation")
    tex += r"""
\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{figures/fig5_feature_ablation.pdf}
\caption{Ablation identifies same-ID timing residuals as the strongest local behavior signal in Test04.}
\label{fig:ablation}
\end{figure}

\subsection{Sensitivity to Aggregation Window}
W10, W20, and W100 behave differently across settings. Shorter windows help Test01 and Test02, while W100 is stronger in Test03 and Test04. The result argues for reporting window sensitivity instead of treating one test-selected window as a universal hyperparameter.

\subsection{Low-False-Alarm Operating Points}
Score-based operating analysis is supplementary. Fig.~\ref{fig:opfpr} compares AUPR and the available fixed-FPR detection evidence on Test04. These metrics are useful for deployment discussions, but they are reported only when the underlying score is available under the same protocol.
\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{figures/fig6_low_fpr_operating.pdf}
\caption{Supplementary Test04 operating evidence shows that AUPR and fixed-FPR recall must be interpreted separately from thresholded F1.}
\label{fig:opfpr}
\end{figure}

\subsection{Diagnostic Rule and Statistical Checks}
We include simple single-signal detectors only as diagnostic checks. They test whether GRAIN-CAN is merely a single threshold over timing or payload change. They are not presented as recent state-of-the-art competitors.

\section{Discussion}
\subsection{What GRAIN-CAN Explains}
The experiments support the design claim that local same-ID behavior should be made explicit before window aggregation. GRAIN-CAN is strongest when raw windows struggle to preserve sparse evidence under vehicle or attack shifts.

\subsection{Failure Modes}
\method{} still needs supervised labels and visible disruptions in timing, payload continuity, or local ID behavior. Attacks that preserve those properties can be missed. The method is also not vehicle-independent: it reduces reliance on absolute distributions but does not remove distribution shift.

\subsection{Comparability to Prior Work}
Prior reported aggregate metrics cannot be directly compared with attack-positive F1 unless the positive class, averaging rule, thresholds, and confusion matrix are available. We therefore compare reproduced local rows under a common metric definition and treat score metrics as supplementary.

\section{Conclusion}
\method{} demonstrates that same-ID local behavior extraction before window aggregation is a useful lightweight design for cross-shift CAN IDS. Its strength is interpretable representation-level robustness; its limitation is that supervised labels and visible timing or payload disruptions are still required. Future work should combine this representation with self-supervised training, signal-level features, DBC-aware variants, online adaptation, and real-time deployment evaluation.

\bibliographystyle{splncs04}
\bibliography{grain_can_framework_refs}
\end{document}
"""
    (OUT / "grain_can_framework_style.tex").write_text(tex, encoding="utf-8")


def write_report():
    text = """# Rewrite Report

## What changed
- Rewrote the paper as a GRAIN-CAN detector / pipeline paper rather than an audit-style evaluation report.
- Removed audit-heavy main-text structure. Window, threshold, metric-availability, and prior-comparability details are no longer main narrative tables.
- Replaced the previous crowded method figure with a clean horizontal detector pipeline.
- Replaced raw CSV-dump figures with five compact publication-style figures: pipeline, mechanism, main CT&T result, raw-window comparison, feature ablation, and supplementary low-FPR operating evidence.
- Kept rule/statistical baselines as diagnostic checks only, not as main competitors.

## New structure
1. Introduction
2. Background and Design Motivation
3. GRAIN-CAN Pipeline
4. Experimental Setup
5. Evaluation
6. Discussion
7. Conclusion

## New main figures
1. GRAIN-CAN pipeline
2. Local evidence dilution mechanism
3. Main CT&T cross-shift results
4. Raw fixed-window versus GRAIN-CAN
5. Feature-group ablation
6. Supplementary low-FPR operating evidence

## New main tables
1. Compared IDS pipelines
2. CT&T shift settings
3. Main CT&T results
4. Raw-window versus GRAIN-CAN
5. Feature-group ablation
6. Pipeline positioning

## Checks
- GRAIN-CAN is written as a supervised feature pipeline, not a neural architecture, rule detector, or evaluation framework.
- ACE-CAN language is not used.
- Defensive disclaimers are moved to Discussion.
- The manuscript keeps attack-positive IDS metrics and score-metric cautions.
"""
    (OUT / "rewrite_report.md").write_text(text, encoding="utf-8")


def main() -> None:
    setup()
    conv, raw, ab, win, supp, main_df = build_tables()
    build_figures(conv, raw, ab, win, supp, main_df)
    write_bib()
    write_latex()
    write_report()
    files = sorted(str(p) for p in OUT.rglob("*") if p.is_file())
    (OUT / "inventory.txt").write_text("\n".join(files) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
