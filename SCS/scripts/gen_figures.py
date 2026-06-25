"""Generate publication-ready figures and LaTeX tables for SCS-CAN paper."""
import sys, json, os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

ROOT = Path("/root/autodl-tmp/scs-can")
OUT = ROOT / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
TAB = ROOT / "results" / "tables"

main = pd.read_csv(TAB / "main_results.csv")
few = pd.read_csv(TAB / "few_label_results.csv")
abl = pd.read_csv(TAB / "ablation_results.csv")
cross = pd.read_csv(TAB / "cross_vehicle_results.csv")

MODEL_ORDER = ["cnn", "lstm", "transformer", "scs_can"]
MODEL_LABELS = {"cnn": "CNN", "lstm": "LSTM", "transformer": "Transformer",
                "scs_can": "SCS-CAN"}
VARIANT_LABELS = {"wo_ssl": "w/o SSL", "full": "Full"}
COLORS = {"cnn": "#7f8c8d", "lstm": "#2980b9", "transformer": "#e67e22",
          "scs_can": "#27ae60"}

# ============================================================
# Figure 1: Main results bar chart (3 datasets)
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=False)
datasets = [("road", "ROAD"), ("hcrl_ch", "HCRL-CH"), ("crysys_subset", "CrySyS")]

for ax, (ds, ds_label) in zip(axes, datasets):
    df = main[main.dataset == ds].copy()
    # Build bars: CNN, LSTM, Transformer, SCS-CAN w/o SSL, SCS-CAN full
    labels = []
    f1s = []
    colors = []
    for m in MODEL_ORDER:
        rows = df[df.model == m]
        if m == "scs_can":
            for v, vl in [("wo_ssl", "w/o SSL"), ("full", "Full")]:
                r = rows[rows.variant == v]
                if len(r):
                    labels.append(f"SCS-CAN\n{vl}")
                    f1s.append(r.iloc[0].f1)
                    colors.append(COLORS[m] if v == "full" else "#2ecc71")
        else:
            if len(rows):
                labels.append(MODEL_LABELS[m])
                f1s.append(rows.iloc[0].f1)
                colors.append(COLORS[m])

    bars = ax.bar(range(len(labels)), f1s, color=colors, edgecolor='white', width=0.7)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_title(ds_label, fontweight='bold')
    ax.set_ylabel("F1 Score" if ax == axes[0] else "")
    ax.set_ylim(0, 1.05)
    for bar, v in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{v:.3f}", ha='center', va='bottom', fontsize=7)

plt.tight_layout()
fig.savefig(OUT / "fig1_main_results.pdf")
fig.savefig(OUT / "fig1_main_results.png")
print("Fig1: main results bar chart", flush=True)

# ============================================================
# Figure 2: Few-label line plot
# ============================================================
fig, ax = plt.subplots(figsize=(5, 3.5))
ratios = sorted(few.label_ratio.unique())
for m in ["transformer", "scs_can"]:
    for v in (["full"] if m == "transformer" else ["wo_ssl", "full"]):
        sub = few[(few.model == m) & (few.variant == v)].sort_values("label_ratio")
        label = MODEL_LABELS[m] if m == "transformer" else f"SCS-CAN {VARIANT_LABELS[v]}"
        ls = "-" if v == "full" else "--"
        marker = "o" if m == "transformer" else ("s" if v == "full" else "^")
        ax.plot(sub.label_ratio, sub.f1, marker=marker, label=label,
                color=COLORS[m], linestyle=ls, markersize=5)

ax.set_xlabel("Label Ratio")
ax.set_ylabel("F1 Score")
ax.set_title("Few-Label Performance (ROAD)")
ax.legend(loc='lower right')
ax.set_xscale('log')
ax.set_xticks(ratios)
ax.set_xticklabels([f"{r:.0%}" if r >= 0.1 else f"{r:.0%}" for r in ratios])
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(OUT / "fig2_few_label.pdf")
fig.savefig(OUT / "fig2_few_label.png")
print("Fig2: few-label line plot", flush=True)

# ============================================================
# Figure 3: Ablation bar chart
# ============================================================
fig, ax = plt.subplots(figsize=(5, 3.5))
abl_variants = ["full", "wo_ssl", "wo_mfm", "wo_ipc", "wo_transition"]
abl_labels = ["Full", "w/o SSL", "w/o MFM", "w/o IPC", "w/o Trans."]
abl_colors = ["#27ae60", "#2ecc71", "#3498db", "#e74c3c", "#9b59b6"]
f1s = []
for v in abl_variants:
    r = abl[abl.variant == v]
    f1s.append(r.iloc[0].f1 if len(r) else 0)

bars = ax.bar(range(len(abl_labels)), f1s, color=abl_colors, edgecolor='white', width=0.6)
ax.set_xticks(range(len(abl_labels)))
ax.set_xticklabels(abl_labels, fontsize=9)
ax.set_ylabel("F1 Score")
ax.set_title("Ablation Study (ROAD)")
ax.set_ylim(min(f1s) - 0.02, max(f1s) + 0.02)
for bar, v in zip(bars, f1s):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f"{v:.4f}", ha='center', va='bottom', fontsize=8)
plt.tight_layout()
fig.savefig(OUT / "fig3_ablation.pdf")
fig.savefig(OUT / "fig3_ablation.png")
print("Fig3: ablation bar chart", flush=True)

# ============================================================
# Figure 4: Cross-vehicle bar chart
# ============================================================
fig, ax = plt.subplots(figsize=(4, 3.5))
cv_models = ["transformer", "scs_can"]
cv_labels_list = []
cv_f1s = []
cv_colors = []
for m in cv_models:
    rows = cross[cross.model == m]
    if m == "scs_can":
        for v, vl in [("wo_ssl", "w/o SSL"), ("full", "Full")]:
            r = rows[rows.variant == v]
            if len(r):
                cv_labels_list.append(f"SCS-CAN\n{vl}")
                cv_f1s.append(r.iloc[0].macro_f1)
                cv_colors.append(COLORS[m] if v == "full" else "#2ecc71")
    else:
        if len(rows):
            cv_labels_list.append(MODEL_LABELS[m])
            cv_f1s.append(rows.iloc[0].macro_f1)
            cv_colors.append(COLORS[m])

bars = ax.bar(range(len(cv_labels_list)), cv_f1s, color=cv_colors, edgecolor='white', width=0.6)
ax.set_xticks(range(len(cv_labels_list)))
ax.set_xticklabels(cv_labels_list, fontsize=9)
ax.set_ylabel("Macro F1")
ax.set_title("Cross-Vehicle (Kia→Sonata)")
ax.set_ylim(0, 0.5)
for bar, v in zip(bars, cv_f1s):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{v:.3f}", ha='center', va='bottom', fontsize=8)
plt.tight_layout()
fig.savefig(OUT / "fig4_cross_vehicle.pdf")
fig.savefig(OUT / "fig4_cross_vehicle.png")
print("Fig4: cross-vehicle bar chart", flush=True)

# ============================================================
# LaTeX Table 1: Main results
# ============================================================
def fmt(v, bold=False):
    s = f"{v:.4f}" if v < 1 else "1.0000"
    return f"\\textbf{{{s}}}" if bold else s

lines = []
lines.append(r"\begin{table}[t]")
lines.append(r"\centering")
lines.append(r"\caption{Main Experiment Results (F1 / AUC)}")
lines.append(r"\label{tab:main}")
lines.append(r"\small")
lines.append(r"\begin{tabular}{l|cc|cc|cc}")
lines.append(r"\toprule")
lines.append(r"\multirow{2}{*}{Model} & \multicolumn{2}{c|}{ROAD} & \multicolumn{2}{c|}{HCRL-CH} & \multicolumn{2}{c}{CrySyS} \\")
lines.append(r" & F1 & AUC & F1 & AUC & F1 & AUC \\")
lines.append(r"\midrule")

ds_list = ["road", "hcrl_ch", "crysys_subset"]
model_rows = [
    ("CNN", "cnn", "full"),
    ("LSTM", "lstm", "full"),
    ("Transformer", "transformer", "full"),
    ("SCS-CAN w/o SSL", "scs_can", "wo_ssl"),
    ("SCS-CAN (Full)", "scs_can", "full"),
]

# Find best F1 per dataset
best_f1 = {}
for ds in ds_list:
    dsm = main[main.dataset == ds]
    best_f1[ds] = dsm.f1.max()

for label, model, variant in model_rows:
    parts = [label]
    for ds in ds_list:
        r = main[(main.dataset == ds) & (main.model == model) & (main.variant == variant)]
        if len(r):
            f1v = r.iloc[0].f1
            aucv = r.iloc[0].auroc
            is_best = abs(f1v - best_f1[ds]) < 1e-6
            parts.append(fmt(f1v, is_best))
            parts.append(fmt(aucv))
        else:
            parts.extend(["--", "--"])
    lines.append(" & ".join(parts) + r" \\")

lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\end{table}")

with open(TAB / "table1_main.tex", "w") as f:
    f.write("\n".join(lines))
print("Table1: main results LaTeX", flush=True)

# ============================================================
# LaTeX Table 2: Ablation
# ============================================================
lines2 = []
lines2.append(r"\begin{table}[t]")
lines2.append(r"\centering")
lines2.append(r"\caption{Ablation Study on ROAD Dataset}")
lines2.append(r"\label{tab:ablation}")
lines2.append(r"\small")
lines2.append(r"\begin{tabular}{lccc}")
lines2.append(r"\toprule")
lines2.append(r"Variant & F1 & AUC & $\Delta$F1 \\")
lines2.append(r"\midrule")

full_f1 = abl[abl.variant == "full"].iloc[0].f1
for v, vl in [("full", "Full Model"), ("wo_ssl", "w/o SSL"), ("wo_mfm", "w/o MFM"),
              ("wo_ipc", "w/o IPC"), ("wo_transition", "w/o Transition")]:
    r = abl[abl.variant == v].iloc[0]
    delta = r.f1 - full_f1
    delta_s = f"+{delta:.4f}" if delta > 0 else f"{delta:.4f}" if delta < 0 else "--"
    bold = v == "full"
    lines2.append(f"{vl} & {fmt(r.f1, bold)} & {fmt(r.auroc)} & {delta_s}" + r" \\")

lines2.append(r"\bottomrule")
lines2.append(r"\end{tabular}")
lines2.append(r"\end{table}")

with open(TAB / "table2_ablation.tex", "w") as f:
    f.write("\n".join(lines2))
print("Table2: ablation LaTeX", flush=True)

# ============================================================
# LaTeX Table 3: Few-label
# ============================================================
lines3 = []
lines3.append(r"\begin{table}[t]")
lines3.append(r"\centering")
lines3.append(r"\caption{Few-Label Performance on ROAD (F1 Score)}")
lines3.append(r"\label{tab:fewlabel}")
lines3.append(r"\small")
lines3.append(r"\begin{tabular}{l" + "c" * len(ratios) + "}")
lines3.append(r"\toprule")
lines3.append("Model & " + " & ".join([f"{r:.0%}" for r in ratios]) + r" \\")
lines3.append(r"\midrule")

for m, v, label in [("transformer","full","Transformer"), ("scs_can","wo_ssl","SCS-CAN w/o SSL"), ("scs_can","full","SCS-CAN Full")]:
    parts = [label]
    for ratio in ratios:
        r = few[(few.model == m) & (few.variant == v) & (abs(few.label_ratio - ratio) < 0.001)]
        if len(r):
            parts.append(f"{r.iloc[0].f1:.4f}")
        else:
            parts.append("--")
    lines3.append(" & ".join(parts) + r" \\")

lines3.append(r"\bottomrule")
lines3.append(r"\end{tabular}")
lines3.append(r"\end{table}")

with open(TAB / "table3_fewlabel.tex", "w") as f:
    f.write("\n".join(lines3))
print("Table3: few-label LaTeX", flush=True)

# ============================================================
# LaTeX Table 4: Cross-vehicle
# ============================================================
lines4 = []
lines4.append(r"\begin{table}[t]")
lines4.append(r"\centering")
lines4.append(r"\caption{Cross-Vehicle Generalization (Kia$\rightarrow$Sonata)}")
lines4.append(r"\label{tab:crossvehicle}")
lines4.append(r"\small")
lines4.append(r"\begin{tabular}{lccc}")
lines4.append(r"\toprule")
lines4.append(r"Model & Macro F1 & AUC & FPR \\")
lines4.append(r"\midrule")

for m, v, label in [("transformer","full","Transformer"), ("scs_can","wo_ssl","SCS-CAN w/o SSL"), ("scs_can","full","SCS-CAN Full")]:
    r = cross[(cross.model == m) & (cross.variant == v)]
    if len(r):
        r = r.iloc[0]
        lines4.append(f"{label} & {r.macro_f1:.4f} & {r.auroc:.4f} & {r.fpr:.4f}" + r" \\")

lines4.append(r"\bottomrule")
lines4.append(r"\end{tabular}")
lines4.append(r"\end{table}")

with open(TAB / "table4_crossvehicle.tex", "w") as f:
    f.write("\n".join(lines4))
print("Table4: cross-vehicle LaTeX", flush=True)

print("\n=== All figures and tables generated ===", flush=True)
print(f"Figures: {list(OUT.glob('*.pdf'))}", flush=True)
print(f"Tables: {list(TAB.glob('*.tex'))}", flush=True)
