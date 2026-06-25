"""Generate additional publication figures for top venue."""
import sys, json, re, os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 10, 'axes.labelsize': 11,
    'axes.titlesize': 12, 'legend.fontsize': 9, 'figure.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05,
})

ROOT = Path("/root/autodl-tmp/scs-can")
OUT = ROOT / "results" / "figures"
TAB = ROOT / "results" / "tables"
LOGS = ROOT / "logs"

main = pd.read_csv(TAB / "main_results.csv")
few = pd.read_csv(TAB / "few_label_results.csv")
abl = pd.read_csv(TAB / "ablation_results.csv")

COLORS = {"cnn": "#7f8c8d", "lstm": "#2980b9", "transformer": "#e67e22",
          "scs_can_wo_ssl": "#2ecc71", "scs_can_full": "#27ae60"}

# ============================================================
# Figure 5: Training convergence curves (from logs)
# ============================================================
def parse_log_epochs(logfile, dataset_key="road"):
    """Extract epoch, val_auc, val_f1 from training logs."""
    epochs, aucs, f1s = [], [], []
    pat = re.compile(r"ep (\d+)/\d+ val_auc=([\d.]+) val_f1=([\d.]+)")
    try:
        text = logfile.read_text()
        for m in pat.finditer(text):
            ep, auc, f1 = int(m.group(1)), float(m.group(2)), float(m.group(3))
            epochs.append(ep)
            aucs.append(auc)
            f1s.append(f1)
    except:
        pass
    return epochs, aucs, f1s

fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

# ROAD convergence
log_map = {
    "Transformer": LOGS / "day1_transformer_road.log",
    "SCS-CAN w/o SSL": LOGS / "day2_scs_can_wo_ssl_road.log",
    "SCS-CAN Full": LOGS / "day3_full_road.log",
}
color_map = {"Transformer": "#e67e22", "SCS-CAN w/o SSL": "#2ecc71", "SCS-CAN Full": "#27ae60"}
ls_map = {"Transformer": "-", "SCS-CAN w/o SSL": "--", "SCS-CAN Full": "-"}

for label, logf in log_map.items():
    eps, aucs, f1s = parse_log_epochs(logf)
    if eps:
        axes[0].plot(eps, f1s, label=label, color=color_map[label],
                     linestyle=ls_map[label], marker='.', markersize=3)
        axes[1].plot(eps, aucs, label=label, color=color_map[label],
                     linestyle=ls_map[label], marker='.', markersize=3)

axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Val F1"); axes[0].set_title("Validation F1 (ROAD)")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Val AUC"); axes[1].set_title("Validation AUC (ROAD)")
for ax in axes:
    ax.legend(loc='lower right'); ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(OUT / "fig5_convergence.pdf")
fig.savefig(OUT / "fig5_convergence.png")
print("Fig5: convergence curves", flush=True)

# ============================================================
# Figure 6: Precision-Recall tradeoff (bar chart)
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
datasets = [("road", "ROAD"), ("hcrl_ch", "HCRL-CH"), ("crysys_subset", "CrySyS")]
model_list = [("CNN", "cnn", "full"), ("LSTM", "lstm", "full"),
              ("Transformer", "transformer", "full"),
              ("SCS-CAN\nw/o SSL", "scs_can", "wo_ssl"),
              ("SCS-CAN\nFull", "scs_can", "full")]

for ax, (ds, ds_label) in zip(axes, datasets):
    labels, precs, recs = [], [], []
    for ml, m, v in model_list:
        r = main[(main.dataset == ds) & (main.model == m) & (main.variant == v)]
        if len(r):
            labels.append(ml)
            precs.append(r.iloc[0].precision)
            recs.append(r.iloc[0].recall)

    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w/2, precs, w, label='Precision', color='#3498db', alpha=0.8)
    ax.bar(x + w/2, recs, w, label='Recall', color='#e74c3c', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_title(ds_label, fontweight='bold')
    ax.set_ylabel("Score" if ax == axes[0] else "")
    ax.set_ylim(0, 1.1)
    if ax == axes[0]:
        ax.legend(loc='upper left', fontsize=8)

plt.tight_layout()
fig.savefig(OUT / "fig6_precision_recall.pdf")
fig.savefig(OUT / "fig6_precision_recall.png")
print("Fig6: precision-recall comparison", flush=True)

# ============================================================
# Figure 7: FPR comparison (critical for IDS)
# ============================================================
fig, ax = plt.subplots(figsize=(6, 3.5))
ds_labels = ["ROAD", "HCRL-CH", "CrySyS"]
ds_keys = ["road", "hcrl_ch", "crysys_subset"]
models_for_fpr = [("Transformer", "transformer", "full", "#e67e22"),
                  ("SCS-CAN w/o SSL", "scs_can", "wo_ssl", "#2ecc71"),
                  ("SCS-CAN Full", "scs_can", "full", "#27ae60")]

x = np.arange(len(ds_labels))
w = 0.25
for i, (ml, m, v, c) in enumerate(models_for_fpr):
    fprs = []
    for ds in ds_keys:
        r = main[(main.dataset == ds) & (main.model == m) & (main.variant == v)]
        fprs.append(r.iloc[0].fpr * 100 if len(r) else 0)
    ax.bar(x + (i - 1) * w, fprs, w, label=ml, color=c)

ax.set_xticks(x)
ax.set_xticklabels(ds_labels)
ax.set_ylabel("FPR (%)")
ax.set_title("False Positive Rate Comparison")
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
fig.savefig(OUT / "fig7_fpr.pdf")
fig.savefig(OUT / "fig7_fpr.png")
print("Fig7: FPR comparison", flush=True)

# ============================================================
# LaTeX Table 5: Dataset statistics
# ============================================================
ds_stats = [
    ("ROAD", "28M", "5 types", "1.89\\%", "Ambient driving + injected attacks"),
    ("HCRL-CH", "18.5M", "3 types", "41.6\\%", "Real CAN bus, multiple ECUs"),
    ("HCRL-SA", "1.9M", "3 types", "27.7\\%", "3 vehicles (Kia, Sonata, Soul)"),
    ("CrySyS (subset)", "5.5M", "2 types", "2.86\\%", "Fabrication + masquerade"),
]

lines = []
lines.append(r"\begin{table}[t]")
lines.append(r"\centering")
lines.append(r"\caption{Dataset Statistics}")
lines.append(r"\label{tab:datasets}")
lines.append(r"\small")
lines.append(r"\begin{tabular}{lcccc}")
lines.append(r"\toprule")
lines.append(r"Dataset & Frames & Attack Types & Attack Rate & Description \\")
lines.append(r"\midrule")
for name, frames, atypes, rate, desc in ds_stats:
    lines.append(f"{name} & {frames} & {atypes} & {rate} & {desc}" + r" \\")
lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\end{table}")

with open(TAB / "table5_datasets.tex", "w") as f:
    f.write("\n".join(lines))
print("Table5: dataset statistics", flush=True)

# ============================================================
# LaTeX Table 6: Full metrics (Accuracy, Precision, Recall, F1, AUC, FPR)
# ============================================================
lines6 = []
lines6.append(r"\begin{table*}[t]")
lines6.append(r"\centering")
lines6.append(r"\caption{Detailed Performance Metrics on ROAD Dataset}")
lines6.append(r"\label{tab:detailed}")
lines6.append(r"\small")
lines6.append(r"\begin{tabular}{lcccccc}")
lines6.append(r"\toprule")
lines6.append(r"Model & Accuracy & Precision & Recall & F1 & AUC & FPR \\")
lines6.append(r"\midrule")

road = main[main.dataset == "road"]
best_f1 = road.f1.max()
for ml, m, v in model_list:
    ml_clean = ml.replace("\n", " ")
    r = road[(road.model == m) & (road.variant == v)]
    if len(r):
        r = r.iloc[0]
        bold = abs(r.f1 - best_f1) < 1e-6
        def f(val, b=False):
            s = f"{val:.4f}"
            return f"\\textbf{{{s}}}" if b else s
        fpr_s = f"{r.fpr*100:.2f}\\%"
        lines6.append(f"{ml_clean} & {f(r.accuracy)} & {f(r.precision)} & {f(r.recall)} & {f(r.f1, bold)} & {f(r.auroc)} & {fpr_s}" + r" \\")

lines6.append(r"\bottomrule")
lines6.append(r"\end{tabular}")
lines6.append(r"\end{table*}")

with open(TAB / "table6_detailed.tex", "w") as f:
    f.write("\n".join(lines6))
print("Table6: detailed metrics", flush=True)

print("\n=== All extra figures and tables generated ===", flush=True)
