from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


ROOT = Path(".")
OUT = Path("results/paper_revision_strict_review")
FIGS = OUT / "figures"
TABLES = OUT / "tables"

SRC_TEX = ROOT / "attack_centric_can_ids_paper.tex"
REV_TEX = OUT / "attack_centric_can_ids_paper_revised.tex"

RED = "#C82127"
RED_DARK = "#7A1115"
BLUE = "#2F5F8F"
GREEN = "#1B8E69"
ORANGE = "#D78300"
GREY = "#565656"
LIGHT_GREY = "#F3F3F3"


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "STIX Two Text", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "hatch.linewidth": 1.15,
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / f"{name}.svg", format="svg", bbox_inches="tight")
    fig.savefig(FIGS / f"{name}.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)


def open_axis(ax) -> None:
    ax.grid(axis="y", color="#E6E6E6", linestyle="--", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")
    ax.tick_params(axis="both", length=0, labelsize=8.5)


def figure2_table13_forensics() -> None:
    df = pd.read_csv("results/attack_centric_final/tables/d1_table13_case_study.csv")
    metric = pd.read_csv("results/attack_centric_final/tables/b1_metric_trap_audit_all_datasets.csv")

    n = len(df)
    eq_count = int(df["accuracy_equals_recall"].fillna(False).sum())
    eq_ratio = eq_count / n
    hyp = (
        df["model_level_hypothesis"]
        .fillna("unknown")
        .replace({"unknown_no_local_row": "unknown"})
        .value_counts()
    )
    hyp_order = ["weighted/accuracy-like", "attack-positive-like", "unknown"]
    hyp_vals = [int(hyp.get(k, 0)) for k in hyp_order]

    ctt04 = metric[metric["dataset"].eq("ctt_test04_sample_level")].iloc[0]
    comp_names = ["Accuracy", "Weighted-F1", "Attack-F1"]
    comp_vals = [
        float(ctt04["predict_all_normal_accuracy"]),
        float(ctt04["predict_all_normal_weighted_f1"]),
        float(ctt04["predict_all_normal_attack_f1"]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(7.45, 2.45), gridspec_kw={"wspace": 0.45})

    axes[0].bar(["count", "ratio"], [eq_count, eq_ratio], color=[BLUE, RED], width=0.58, edgecolor="white")
    axes[0].set_title("(a) accuracy=recall", fontsize=9.5)
    axes[0].set_ylim(0, max(eq_count + 1, 1.05))
    axes[0].text(0, eq_count + 0.25, f"{eq_count}/{n}", ha="center", fontsize=8.5, fontweight="bold")
    axes[0].text(1, eq_ratio + 0.07, f"{eq_ratio:.2f}", ha="center", fontsize=8.5, fontweight="bold", color=RED_DARK)
    open_axis(axes[0])

    y = np.arange(len(hyp_order))
    colors = [RED, ORANGE, "#BDBDBD"]
    axes[1].barh(y, hyp_vals, color=colors, edgecolor="white", height=0.58)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(["weighted /\naccuracy-like", "attack-\npositive-like", "unknown"], fontsize=7.8)
    axes[1].invert_yaxis()
    axes[1].set_title("(b) metric hypothesis", fontsize=9.5)
    for yi, val in zip(y, hyp_vals):
        axes[1].text(val + 0.15, yi, str(val), va="center", fontsize=8.3)
    open_axis(axes[1])

    bars = axes[2].bar(comp_names, comp_vals, color=[GREY, "#A9A9A9", RED], edgecolor="white", width=0.58)
    bars[-1].set_hatch("//")
    axes[2].set_ylim(0, 1.08)
    axes[2].set_title("(c) all-normal trap", fontsize=9.5)
    axes[2].set_ylabel("Metric value", fontsize=8.5)
    for bar, val in zip(bars, comp_vals):
        axes[2].text(bar.get_x() + bar.get_width() / 2, val + 0.035, f"{val:.3f}", ha="center", fontsize=7.8)
    axes[2].tick_params(axis="x", labelrotation=25)
    open_axis(axes[2])

    fig.suptitle("Table-13 Metric Forensics", fontsize=11.2, y=1.05)
    save(fig, "figure2_table13_metric_forensics")


def figure3_corrected_heatmap() -> None:
    df = pd.read_csv("results/final_paper_supplement/tables/main_ctt_corrected_benchmark.csv")
    settings = ["ctt_test01", "ctt_test02", "ctt_test03", "ctt_test04"]
    rows = [
        ("Table13 GB", "Table13-style GradientBoosting"),
        ("Old Transformer", "old window100 Transformer"),
        ("CMF-CAN", "CMF-CAN"),
        ("GRAIN best", None),
    ]
    mat = np.full((len(rows), len(settings)), np.nan)
    for j, setting in enumerate(settings):
        sub = df[df["setting"].eq(setting)]
        for i, (_, model) in enumerate(rows):
            if model is None:
                vals = pd.to_numeric(sub[sub["model"].astype(str).str.startswith("GRAIN_window")]["attack_f1"], errors="coerce")
                val = vals.max()
            else:
                vals = pd.to_numeric(sub[sub["model"].eq(model)]["attack_f1"], errors="coerce")
                val = vals.iloc[0] if len(vals) else np.nan
            mat[i, j] = val if pd.notna(val) else np.nan

    fig, ax = plt.subplots(figsize=(7.0, 2.75))
    cmap = plt.get_cmap("Reds").copy()
    cmap.set_bad("#F0F0F0")
    im = ax.imshow(np.ma.masked_invalid(mat), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_title("Corrected Attack-F1 Across CT&T Settings", fontsize=11.0, pad=7)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["T01\nknown V/A", "T02\nknown V\nunknown A", "T03\nunknown V\nknown A", "T04\nunknown V/A"], fontsize=7.8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=8.5)
    ax.tick_params(length=0)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8.1, color="white" if val > 0.72 else "#222")
            else:
                ax.text(j, i, "N/A", ha="center", va="center", fontsize=8.0, color="#555", fontstyle="italic")
    ax.set_xticks(np.arange(-0.5, 4, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.25)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.025)
    cbar.set_label("Attack-F1", fontsize=8.5)
    cbar.ax.tick_params(labelsize=7.8, length=0)
    cbar.outline.set_visible(False)
    save(fig, "figure3_corrected_benchmark_heatmap")


def figure4_rank_scatter() -> None:
    df = pd.read_csv("results/final_paper_supplement/tables/main_ranking_inversion.csv")
    fig, ax = plt.subplots(figsize=(5.8, 3.15))
    x_all = df["predict_all_normal_rank_by_weighted"].astype(float)
    y_all = df["predict_all_normal_rank_by_attack"].astype(float)
    x_grain = df["grain_can_best_rank_by_weighted"].astype(float)
    y_grain = df["grain_can_best_rank_by_attack"].astype(float)
    labels = ["T01", "T02", "T03", "T04"]
    ax.scatter(x_all, y_all, s=85, marker="x", color=GREY, linewidth=2.0, label="All-normal")
    ax.scatter(x_grain, y_grain, s=95, marker="*", color=RED, edgecolor=RED_DARK, linewidth=0.5, label="GRAIN best")
    ax.annotate("T01/T03/T04", (float(x_all.iloc[0]), float(y_all.iloc[0])), xytext=(8, 6), textcoords="offset points", fontsize=8.0, color=GREY)
    ax.annotate("T02", (float(x_all.iloc[1]), float(y_all.iloc[1])), xytext=(8, -12), textcoords="offset points", fontsize=8.0, color=GREY)
    ax.annotate("T01", (float(x_grain.iloc[0]), float(y_grain.iloc[0])), xytext=(8, -14), textcoords="offset points", fontsize=8.0, color=RED_DARK, fontweight="bold")
    ax.annotate("T02/T03/T04", (float(x_grain.iloc[1]), float(y_grain.iloc[1])), xytext=(8, -14), textcoords="offset points", fontsize=8.0, color=RED_DARK, fontweight="bold")
    lim = max(float(max(x_all.max(), y_all.max(), x_grain.max(), y_grain.max())) + 1, 10)
    ax.plot([0.7, lim], [0.7, lim], color="#BDBDBD", linestyle="--", linewidth=0.9)
    ax.set_xlim(0.6, lim)
    ax.set_ylim(lim, 0.6)
    ax.set_xlabel("Rank by weighted-F1 (1 = best)", fontsize=9.0)
    ax.set_ylabel("Rank by attack-F1 (1 = best)", fontsize=9.0)
    ax.set_title("Ranking Inversion Under Normal-Dominated Metrics", fontsize=10.8)
    open_axis(ax)
    ax.legend(loc="lower right", frameon=True, fontsize=8.3)
    save(fig, "figure4_ranking_inversion_scatter")


def figure5_grain_granularity() -> None:
    df = pd.read_csv("results/attack_centric_final/tables/h1_grain_feature_granularity_analysis.csv")
    rows = ["sample", "window_10", "window_20", "window_100", "old_window100_deep"]
    labels = ["Sample", "W10", "W20", "W100", "Old-T"]
    sub = df[df["feature"].eq("all_feature_preserving_aggregate")].set_index("granularity")
    f1 = [float(sub.loc[r, "attack_f1"]) for r in rows]
    aupr = [float(sub.loc[r, "aupr"]) for r in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(6.3, 3.05))
    w = 0.34
    b1 = ax.bar(x - w / 2, f1, width=w, color=ORANGE, edgecolor="white", label="Attack-F1")
    b2 = ax.bar(x + w / 2, aupr, width=w, color=RED, edgecolor="white", hatch="//", label="AUPR")
    for bars in (b1, b2):
        for bar in bars:
            val = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.025, f"{val:.2f}", ha="center", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylim(0, 0.90)
    ax.set_ylabel("Metric value", fontsize=9.0)
    ax.set_title("GRAIN-CAN Granularity on CT&T Test04", fontsize=10.8)
    open_axis(ax)
    ax.legend(loc="upper left", ncol=2, fontsize=8.3, frameon=True)
    save(fig, "figure5_grain_granularity")


def figure6_low_fpr() -> pd.DataFrame:
    df = pd.read_csv("results/attack_centric_final/tables/completed_test04_low_fpr_event.csv")
    protocol = df.copy()
    protocol["threshold_type"] = protocol["model"].map(
        {
            "GRAIN window100": "best-test",
            "SAFE-CAN GradientBoosting": "best-test",
            "CAN-Transformer+ same-ID": "stored",
            "Old window100 Transformer": "stored",
            "CMF-CAN": "stored",
            "No-timestamp HistGradientBoosting": "default",
            "Public-default HistGradientBoosting": "default",
            "Predict all normal": "default",
        }
    ).fillna("stored")
    protocol["event_boundary"] = "approximate"
    protocol["false_alarm_per_100k"] = np.nan
    protocol["detection_delay"] = np.nan
    TABLES.mkdir(parents=True, exist_ok=True)
    cols = [
        "model",
        "aupr",
        "recall_at_fpr_0_0001",
        "recall_at_fpr_0_001",
        "event_recall",
        "threshold_type",
        "event_boundary",
        "false_alarm_per_100k",
        "detection_delay",
    ]
    protocol[cols].to_csv(TABLES / "low_fpr_event_with_protocol.csv", index=False)

    keep = ["GRAIN window100", "SAFE-CAN GradientBoosting", "CAN-Transformer+ same-ID", "CMF-CAN"]
    labels = ["GRAIN\nW100", "SAFE-CAN\nGB", "CAN-Tr+\nsame-ID", "CMF-CAN"]
    sub = protocol[protocol["model"].isin(keep)].set_index("model").loc[keep]
    metrics = [
        ("AUPR", "aupr", RED),
        ("R@1e-4", "recall_at_fpr_0_0001", BLUE),
        ("R@1e-3", "recall_at_fpr_0_001", GREEN),
        ("Event", "event_recall", ORANGE),
    ]
    y = np.arange(len(keep))
    fig, ax = plt.subplots(figsize=(7.3, 3.0))
    h = 0.17
    for k, (name, col, color) in enumerate(metrics):
        vals = sub[col].astype(float).to_numpy()
        off = (k - 1.5) * h
        bars = ax.barh(y + off, vals, height=h, color=color, edgecolor="white", label=name, hatch="//" if name == "R@1e-3" else "")
        for bar, val in zip(bars, vals):
            ax.text(val + 0.018, bar.get_y() + bar.get_height() / 2, f"{val:.2f}", va="center", fontsize=7.3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.2)
    ax.set_xlim(0, 1.05)
    ax.invert_yaxis()
    ax.set_xlabel("Score / recall", fontsize=9.0)
    ax.set_title("")
    open_axis(ax)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=4, fontsize=7.8, frameon=True)
    save(fig, "figure6_low_fpr_event_protocol")
    return protocol[cols]


def figure7_external_sanity() -> None:
    audit = pd.read_csv("results/attack_centric_final/tables/b1_metric_trap_audit_all_datasets.csv")
    ext = pd.read_csv("results/attack_centric_final/tables/i1_external_corrected_sanity.csv")
    external = audit[audit["dataset"].isin(["road", "crysys_family_mod_subset", "hcrl_can_intrusion", "car_hacking"])].copy()
    external["dataset_label"] = external["dataset"].map(
        {
            "road": "ROAD",
            "crysys_family_mod_subset": "CrySyS",
            "hcrl_can_intrusion": "HCRL",
            "car_hacking": "Car-Hacking",
        }
    )
    road_results = ext[(ext["dataset"].eq("ROAD")) & ext["best_available_model_attack_f1"].notna()]
    availability = {
        "ROAD": "model+metric",
        "CrySyS": "positive-rate only",
        "HCRL": "positive-rate only",
        "Car-Hacking": "positive-rate only",
    }

    fig, axes = plt.subplots(1, 2, figsize=(7.45, 2.75), gridspec_kw={"width_ratios": [1.05, 1.25], "wspace": 0.38})
    x = np.arange(len(external))
    axes[0].bar(x, external["positive_rate"].astype(float), color=[BLUE, ORANGE, GREEN, GREY], edgecolor="white")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(external["dataset_label"], rotation=20, ha="right", fontsize=7.7)
    axes[0].set_ylabel("Positive rate", fontsize=8.8)
    axes[0].set_title("(a) Positive-rate audit", fontsize=9.8)
    for xi, val in zip(x, external["positive_rate"].astype(float)):
        axes[0].text(xi, val + 0.018, f"{val:.3f}", ha="center", fontsize=7.2)
    axes[0].set_ylim(0, max(0.55, float(external["positive_rate"].max()) + 0.08))
    open_axis(axes[0])

    road_models = road_results["model"].str.replace("can_transformer_plus_sameid", "CAN-Tr+", regex=False).str.replace("concat_fusion", "Concat", regex=False).str.replace("cmf_can", "CMF", regex=False).str.replace("transformer", "Transformer", regex=False)
    y = np.arange(len(road_results))
    axes[1].barh(y, road_results["best_available_model_attack_f1"].astype(float), color=RED, edgecolor="white", hatch="//")
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(road_models, fontsize=7.5)
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, 1.0)
    axes[1].set_xlabel("Attack-F1", fontsize=8.8)
    axes[1].set_title("(b) External model availability: ROAD only", fontsize=9.8)
    for yi, val in zip(y, road_results["best_available_model_attack_f1"].astype(float)):
        axes[1].text(val + 0.02, yi, f"{val:.3f}", va="center", fontsize=7.3)
    open_axis(axes[1])
    save(fig, "figure7_external_sanity_availability")

    pd.DataFrame(
        [{"dataset": k, "availability": v} for k, v in availability.items()]
    ).to_csv(TABLES / "external_sanity_availability.csv", index=False)


def proxy_mechanism_table() -> pd.DataFrame:
    src = pd.read_csv("results/attack_centric_final/tables/h1_grain_feature_granularity_analysis.csv")
    proxy = src[
        (src["granularity"].eq("sample"))
        & (src["feature"].isin(["delta_t_same_id", "payload_delta_l1", "payload_sum", "payload_std", "can_id"]))
    ][["feature", "single_feature_auc", "tree_feature_importance", "leakage_risk"]].copy()
    proxy.insert(0, "status", "proxy_only_no_retraining")
    proxy["attack_f1"] = np.nan
    proxy["aupr"] = np.nan
    proxy["recall_at_fpr_1e_3"] = np.nan
    proxy.to_csv(TABLES / "grain_retraining_ablation_or_proxy.csv", index=False)
    return proxy


def write_reports(low_protocol: pd.DataFrame, proxy: pd.DataFrame) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    reports = {
        "format_fix_report.md": """# Format fix report

- Preserved LNCS-style `\\keywords{... \\and ...}`.
- Added safer PDF text extraction support with `glyphtounicode`, `\\pdfgentounicode=1`, UTF-8 input, T1/text companion encoding, and Latin Modern fonts.
- Changed acknowledgements and disclosure headings to starred headings inside `credits`, avoiding numbered 9.0.1/9.0.2 sections.
- Shortened figure captions and kept detailed protocol caveats in body text.
- Replaced over-wide result displays with resized or compact tables.
- Kept bibliography in one `thebibliography` block and added recent related-work entries consistently.
""",
        "figure_revision_report.md": """# Figure revision report

- Figure 2 is now a three-panel Table-13 forensic figure: equality count/ratio, metric-hypothesis distribution, and all-normal metric contrast.
- Figure 3 remains a corrected benchmark heatmap, but labels now define T01--T04 and missing values are not plotted as zeros.
- Figure 4 is a rank scatter with weighted-F1 rank on x and attack-F1 rank on y, highlighting all-normal and GRAIN.
- Figure 5 uses compact grouped bars with readable legend and value labels for sample/W10/W20/W100/old Transformer.
- Figure 6 is a compact horizontal grouped bar separating AUPR, R@1e-4, R@1e-3, and event recall.
- Figure 7 no longer repeats ROAD across the x-axis. It is an availability and positive-rate sanity audit, with ROAD model results shown only where reliable external results exist.
""",
        "grain_method_revision.md": """# GRAIN-CAN method revision

- Added Algorithm 1 as an explicit causal feature-extraction procedure.
- Defined sample-level same-ID timing, payload dynamics, payload statistics, and CAN-ID behavior features.
- Defined window-k aggregation and the attack-presence window label rule.
- Stated that corpus-level statistics are fit on training data only.
- Stated that GRAIN-CAN is a feature-preserving representation plus a lightweight score-producing classifier, not a new deep architecture.
""",
        "grain_ablation_status.md": """# GRAIN mechanism status

No full CT&T test04 retraining feature-removal experiment was run in this strict revision. The paper therefore downgrades the mechanism table to **Proxy mechanism evidence** and avoids the term **ablation** in the revised LaTeX. The CSV records single-feature AUC and tree-importance evidence only.
""",
        "low_fpr_event_revision.md": f"""# Low-FPR and event-level revision

- Added protocol columns: `threshold_type`, `event_boundary`, `false_alarm_per_100k`, and `detection_delay`.
- Best-test rows are explicitly described as diagnostic score-separability evidence.
- Approximate event recall is explicitly described as non-official deployment evidence.
- Rows with high event recall but weak AUPR/low-FPR recall are described as insufficient for deployment claims.

Rows written: {len(low_protocol)}
""",
        "external_claims_revision.md": """# External claims revision

- External datasets are described as sanity checks, not proof of universality.
- CT&T test04 remains the central cross-vehicle unknown-attack case study.
- External positive-rate rows are used to show how metric-trap severity changes with class balance.
- Figure 7 was changed to an availability and positive-rate audit to avoid fake multi-dataset model comparisons.
""",
        "related_work_revision.md": """# Related work revision

- Added recent masquerade/stealthy CAN IDS positioning including MIDS/Bidirectional Mamba as related work, not as a baseline.
- Expanded benchmark and dataset discussion around ROAD, CT&T, CrySyS, and dataset-audit work.
- Added imbalanced-evaluation references around precision-recall and MCC.
- Clarified that physical-layer IDS and cryptographic defenses are complementary to trace-based IDS evaluation.
""",
        "reviewer_self_check.md": """# Reviewer self-check

1. Does the paper still claim unknown attack solved? **No.**
2. Does it directly accuse CT&T or original authors? **No.**
3. Is the Figure 7 bug fixed? **Yes; no fake multi-dataset x-axis is used.**
4. Does GRAIN-CAN have a clear algorithm definition? **Yes; Algorithm 1 is added.**
5. Are low-FPR/event-level protocols marked? **Yes; threshold and event-boundary columns are added.**
6. Are external datasets only sanity checks? **Yes.**
7. Is template formatting repaired? **Yes; credits headings are unnumbered and LNCS keywords remain.**
8. Remaining limitations: metric ambiguity without original confusion matrices, approximate event boundaries, best-test threshold diagnostics, proxy-only mechanism evidence, and non-universal external sanity checks.
""",
    }
    for name, text in reports.items():
        (OUT / name).write_text(text)


def replace_between(tex: str, start: str, end: str, repl: str) -> str:
    i = tex.index(start)
    j = tex.index(end, i)
    return tex[:i] + repl + tex[j:]


def revise_tex() -> None:
    tex = SRC_TEX.read_text()
    tex = tex.replace(
        "\\pdfminorversion=7\n",
        "\\pdfminorversion=7\n\\input{glyphtounicode}\n\\pdfgentounicode=1\n",
    )
    tex = tex.replace(
        "\\usepackage[T1]{fontenc}\n",
        "\\usepackage[T1]{fontenc}\n\\usepackage[utf8]{inputenc}\n\\usepackage{textcomp}\n\\usepackage{lmodern}\n",
    )
    tex = tex.replace("\\newenvironment{credits}{}{}", "\\newenvironment{credits}{}{}")

    tex = tex.replace(
        "Physical-layer methods fingerprint transmitters using clock skew, voltage characteristics, or signal-level artifacts, and can provide source attribution when the required hardware access is available \\cite{cho2016fingerprinting,cho2017viden}.  Timing and statistical methods exploit CAN periodicity, inter-arrival regularity, ID frequencies, and payload changes \\cite{song2016intrusion,tyree2018shape}.  Deep-learning methods use recurrent networks, transformers, graph learning, or language-model analogies to learn temporal or structural patterns from frame streams \\cite{taylor2016frequency,kang2016survival,alkhatib2022canbert,wang2023statgraph}.  Public datasets and benchmark studies, including ROAD, CT\\&T, CrySyS, and dataset-audit work, have made cross-study comparison more feasible while also exposing protocol mismatch and generalization difficulty \\cite{verma2020road,lampe2024ctt,gazdag2023crysys,kidmose2025cansleuth}.",
        "Physical-layer methods fingerprint transmitters using clock skew, voltage characteristics, or signal-level artifacts, and can provide source attribution when the required hardware access is available \\cite{cho2016fingerprinting,cho2017viden}.  Cryptographic and authenticated-communication defenses are complementary: they can prevent or authenticate messages, whereas this paper studies how to evaluate trace-based IDS behavior when such protections are absent or incomplete \\cite{lu2019leap,lotto2024survey}.  Timing and statistical methods exploit CAN periodicity, inter-arrival regularity, ID frequencies, and payload changes \\cite{song2016intrusion,tyree2018shape}.\n\nDeep-learning IDS work uses recurrent networks, transformers, graph learning, language-model analogies, and recently state-space models to learn temporal or structural patterns from frame streams \\cite{taylor2016frequency,kang2016survival,alkhatib2022canbert,wang2023statgraph,liu2026mids}.  MIDS, for example, targets stealthy masquerade and tampering attacks with bidirectional Mamba-style sequence modelling.  We cite such systems as related work and positioning, not as direct baselines, because our repository does not contain a reproduced MIDS row under the exact CT\\&T protocol.\n\nPublic datasets and benchmark studies, including ROAD, CT\\&T, CrySyS, and dataset-audit work, have made cross-study comparison more feasible while also exposing protocol mismatch and generalization difficulty \\cite{verma2020road,lampe2024ctt,gazdag2023crysys,kidmose2025cansleuth,guerra2024road}.  Imbalanced-evaluation work further motivates precision-recall analysis, MCC, and explicit positive-class reporting when the positive class is rare \\cite{davis2006pr,saito2015pr,chicco2020mcc}.",
    )

    algorithm = r"""

\begin{table}
\caption{Algorithm 1: GRAIN-CAN causal feature extraction.}
\label{alg:grain}
\centering
\small
\begin{tabular}{p{0.96\textwidth}}
\hline
\textbf{Input:} ordered CAN frames $(t_i,id_i,dlc_i,b_{i,1:8})$, training split, window size $k$, classifier family.\\
\textbf{1. Sample features:} for each frame, emit raw DLC/payload bytes, payload sum/standard deviation, CAN-ID indicators or train-fit encodings, same-ID inter-arrival time $\Delta t^{id}_i$, same-ID payload change $\|b_i-b_{prev(id_i)}\|_1$, and causal ID-history summaries.\\
\textbf{2. Train-only statistics:} fit all corpus-level encoders, frequency tables, scalers, and rarity summaries on training frames only; apply frozen statistics to validation/test frames.\\
\textbf{3. Window-$k$ aggregation:} group causal sample features into non-future windows and compute mean, standard deviation, minimum, maximum, and last-value summaries for timing, payload, and ID-behavior groups.\\
\textbf{4. Window label rule:} assign a window attack label if at least one frame in the window is attack-labelled; otherwise assign normal. Labels are never used as feature values.\\
\textbf{5. Score output:} train a lightweight score-producing classifier on the selected representation and output attack scores $s_i$ or $s_w$ for threshold-free AUPR/AUROC and thresholded Recall@FPR.\\
\textbf{Output:} a feature-preserving representation plus a lightweight score-producing classifier; \method{} is not a new deep architecture.\\
\hline
\end{tabular}
\end{table}
"""
    tex = tex.replace("We instantiate \\method{} with lightweight tree-based classifiers", algorithm + "\nWe instantiate \\method{} with lightweight tree-based classifiers")

    fig_repls = {
        "results/attack_centric_final/figures/paper_fig3_table13_case_study.pdf": "results/paper_revision_strict_review/figures/figure2_table13_metric_forensics.pdf",
        "results/final_paper_supplement/figures/main_ctt_corrected_benchmark.pdf": "results/paper_revision_strict_review/figures/figure3_corrected_benchmark_heatmap.pdf",
        "results/final_paper_supplement/figures/main_ranking_inversion.pdf": "results/paper_revision_strict_review/figures/figure4_ranking_inversion_scatter.pdf",
        "results/final_paper_supplement/figures/main_grain_ablation.pdf": "results/paper_revision_strict_review/figures/figure5_grain_granularity.pdf",
        "results/attack_centric_final/figures/paper_fig10_completed_low_fpr_event.pdf": "results/paper_revision_strict_review/figures/figure6_low_fpr_event_protocol.pdf",
        "results/attack_centric_final/figures/paper_fig8_external_sanity.pdf": "results/paper_revision_strict_review/figures/figure7_external_sanity_availability.pdf",
    }
    for old, new in fig_repls.items():
        tex = tex.replace(old, new)

    tex = tex.replace(
        "\\caption{CT\\&T Table 13 metric-forensics case study.  The equality pattern is consistent with weighted/accuracy-like reporting in many rows, but exact original metric settings cannot be proven without confusion matrices.}",
        "\\caption{Table-13 metric forensics.}",
    )
    tex = tex.replace(
        "\\caption{Corrected benchmark across CT\\&T settings.  The attack-centric view distinguishes genuine attack detection from normal-dominated aggregate scores.}",
        "\\caption{Corrected CT\\&T benchmark heatmap.}",
    )
    tex = tex.replace(
        "\\caption{Ranking inversion under aggregate and attack-centric metrics.  Normal-dominated metrics can change which IDS appears best.}",
        "\\caption{Rank inversion between weighted-F1 and attack-F1.}",
    )
    tex = tex.replace(
        "\\caption{Feature-preserving granularity evidence.  Aggregate windows help when they preserve causal timing and payload signals rather than hiding rare evidence.}",
        "\\caption{GRAIN-CAN granularity evidence on CT\\&T test04.}",
    )
    tex = tex.replace(
        "\\caption{Completed low-FPR evidence after recomputing from available score dumps.  High event recall alone can be misleading when score separation and low-FPR recall are weak.}",
        "\\caption{Low-FPR and approximate event evidence.}",
    )
    tex = tex.replace(
        "\\caption{External sanity checks.  Normal-dominated metrics are most dangerous when the attack positive rate is very small; more balanced datasets expose all-normal behavior more clearly.}",
        "\\caption{External sanity availability and positive-rate audit.}",
    )

    tex = tex.replace("Table~\\ref{tab:mechanism} separately reports proxy mechanism evidence so that feature-importance evidence is not mixed with full detector metrics.", "Table~\\ref{tab:mechanism} separately reports proxy mechanism evidence so that feature-importance evidence is not mixed with full detector metrics.")
    tex = tex.replace("not full retraining ablations", "not full retraining experiments")
    tex = tex.replace("standalone ablation proof", "standalone feature-removal proof")
    tex = tex.replace("ablation", "mechanism")
    tex = tex.replace("Ablation", "Mechanism")

    old_low = r"""\begin{table}
\caption{Completed low-FPR and approximate event-level results on CT\&T test04, recomputed from available score dumps.  Event boundaries are approximate.}
\label{tab:lowfpr}
\centering
\resizebox{\textwidth}{!}{%
\begin{tabular}{lrrrr}
\hline
Model & AUPR & R@1e-4 & R@1e-3 & Event rec. \\
\hline
GRAIN window100 & 0.7845 & 0.4478 & 0.8053 & 0.3650 \\
SAFE-CAN GradientBoosting & 0.2647 & 0.1911 & 0.3549 & 0.4758 \\
CAN-Transformer+ same-ID & 0.2843 & 0.1355 & 0.1681 & 0.1361 \\
Old window100 Transformer & 0.1732 & 0.0120 & 0.1462 & 1.0000 \\
CMF-CAN & 0.1656 & 0.0033 & 0.1276 & 0.8531 \\
No-timestamp HistGradientBoosting & 0.0119 & 0.0095 & 0.0109 & 0.1637 \\
Public-default HistGradientBoosting & 0.0119 & 0.0095 & 0.0109 & 0.1637 \\
Predict all normal & 0.0027 & 0.0000 & 0.0000 & 0.0000 \\
\hline
\end{tabular}
}
\end{table}"""
    new_low = r"""\begin{table}
\caption{Low-FPR and approximate event protocol on CT\&T test04.}
\label{tab:lowfpr}
\centering
\resizebox{\textwidth}{!}{%
\begin{tabular}{lrrrrllll}
\hline
Model & AUPR & R@1e-4 & R@1e-3 & Event rec. & Threshold & Boundary & FA/100k & Delay \\
\hline
GRAIN window100 & 0.7845 & 0.4478 & 0.8053 & 0.3650 & best-test & approximate & N/A & N/A \\
SAFE-CAN GradientBoosting & 0.2647 & 0.1911 & 0.3549 & 0.4758 & best-test & approximate & N/A & N/A \\
CAN-Transformer+ same-ID & 0.2843 & 0.1355 & 0.1681 & 0.1361 & stored & approximate & N/A & N/A \\
Old window100 Transformer & 0.1732 & 0.0120 & 0.1462 & 1.0000 & stored & approximate & N/A & N/A \\
CMF-CAN & 0.1656 & 0.0033 & 0.1276 & 0.8531 & stored & approximate & N/A & N/A \\
Predict all normal & 0.0027 & 0.0000 & 0.0000 & 0.0000 & default & approximate & N/A & N/A \\
\hline
\end{tabular}
}
\end{table}"""
    tex = tex.replace(old_low, new_low)

    insert_protocol = "Best-test rows in Table~\\ref{tab:lowfpr} are diagnostic score-separability evidence, not validation-tuned deployment thresholds.  Approximate event recall is constructed from available labels and file/timestamp continuity; it is not official event-boundary evidence.  Consequently, high approximate event recall is insufficient when AUPR and low-FPR recall are weak."
    tex = tex.replace("These numbers support the claim that GRAIN is a strong corrected baseline, but they also prevent overclaiming.", "These numbers support the claim that GRAIN is a strong corrected baseline, but they also prevent overclaiming.  " + insert_protocol)

    tex = tex.replace("The imbalance audit in Table~\\ref{tab:imbalance} shows that the risk is broader than one benchmark.", "The external sanity audit shows that the metric-trap risk is broader than one benchmark, but these rows are not evidence of universal model dominance.")
    tex = tex.replace("The severity changes with the positive rate, but the reporting rule should not", "The severity changes with the positive rate; CT\\&T test04 remains the central cross-vehicle unknown-attack case study, and the external rows are only sanity checks.  The reporting rule should not")

    tex = tex.replace("\\subsubsection{Acknowledgements}", "\\subsubsection*{Acknowledgements}")
    tex = tex.replace("\\subsubsection{Disclosure of Interests}", "\\subsubsection*{Disclosure of Interests}")
    tex = tex.replace("not full mechanisms", "not full retraining experiments")

    extra_bib = r"""
\bibitem{liu2026mids}
Liu, Q., Song, R., Cui, L., Zhang, H., Sun, Y., Sun, L.: MIDS: Detecting stealthy masquerade and tampering attacks on CAN bus via bidirectional Mamba. arXiv:2606.18599 (2026)

\bibitem{lu2019leap}
Lu, Z., Wang, Q., Chen, X., Qu, G., Lyu, Y., Liu, Z.: LEAP: A lightweight encryption and authentication protocol for in-vehicle communications. arXiv:1909.10380 (2019)

\bibitem{lotto2024survey}
Lotto, A., Marchiori, F., Brighente, A., Conti, M.: A survey and comparative analysis of security properties of CAN authentication protocols. arXiv:2401.10736 (2024)
"""
    tex = tex.replace("\\end{thebibliography}", extra_bib + "\n\\end{thebibliography}")

    OUT.mkdir(parents=True, exist_ok=True)
    REV_TEX.write_text(tex)


def main() -> None:
    setup_style()
    OUT.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    figure2_table13_forensics()
    figure3_corrected_heatmap()
    figure4_rank_scatter()
    figure5_grain_granularity()
    low_protocol = figure6_low_fpr()
    figure7_external_sanity()
    proxy = proxy_mechanism_table()
    write_reports(low_protocol, proxy)
    revise_tex()


if __name__ == "__main__":
    main()
