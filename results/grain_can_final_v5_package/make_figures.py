from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

OUT = Path('/mnt/data/grain_can_final_v3/figures')
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 8,
    'axes.titlesize': 9,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.linewidth': 0.6,
    'grid.linewidth': 0.4,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

# Colors (restrained conference palette)
colors = {
    'blue': '#2F5D8C',
    'teal': '#2A9D8F',
    'orange': '#D99032',
    'red': '#B85C5C',
    'gray': '#6C757D',
    'lightgray': '#E9ECEF',
    'green': '#52796F',
    'purple': '#6D597A',
}

def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f'{name}.pdf', bbox_inches='tight')
    fig.savefig(OUT / f'{name}.svg', bbox_inches='tight')
    plt.close(fig)

# Figure 1: pipeline
fig, ax = plt.subplots(figsize=(7.0, 2.0))
ax.set_axis_off()
boxes = [
    ('Raw CAN\nstream', 'timestamp, ID,\nDLC, payload'),
    ('Per-ID\nhistory state', 'last time,\nlast payload,\nrecent counts'),
    ('Local behavior\nresiduals', 'time gap,\npayload change,\nID behavior'),
    ('Fixed-window\naggregation', 'mean, max,\nstd, last'),
    ('Supervised\ndetector', 'score + label')
]
x0s = np.linspace(0.02, 0.82, len(boxes))
y = 0.43
w, h = 0.145, 0.42
facecolors = ['#F4F7FA', '#EEF6FA', '#EEF8F6', '#FDF7ED', '#F8EFF0']
edgecolors = [colors['blue'], colors['blue'], colors['teal'], colors['orange'], colors['red']]
for i, ((title, body), x) in enumerate(zip(boxes, x0s)):
    patch = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.018,rounding_size=0.02',
                           linewidth=0.9, edgecolor=edgecolors[i], facecolor=facecolors[i])
    ax.add_patch(patch)
    ax.text(x+w/2, y+h*0.66, title, ha='center', va='center', fontsize=8, fontweight='bold')
    ax.text(x+w/2, y+h*0.31, body, ha='center', va='center', fontsize=7)
    if i < len(boxes)-1:
        ax.add_patch(FancyArrowPatch((x+w+0.01, y+h/2), (x0s[i+1]-0.01, y+h/2),
                                     arrowstyle='-|>', mutation_scale=9, linewidth=0.9,
                                     color='#495057'))
ax.text(0.02, 0.92, 'GRAIN-CAN detector pipeline', fontsize=10, fontweight='bold', ha='left')
ax.text(0.02, 0.08, 'Feature definitions, window size, detector, and threshold are fixed before test-time evaluation.', fontsize=7, ha='left', color='#495057')
save(fig, 'fig1_pipeline')

# Figure 2: mechanism
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.05), sharey=True)
for ax in axes:
    ax.spines[['top','right','left']].set_visible(False)
    ax.set_yticks([])
    ax.set_ylim(0, 1)
    ax.set_xlabel('Frames in one window')
    ax.grid(axis='y', alpha=.2)
frames = np.arange(100)
raw = np.ones(100)*0.16
raw[46:52] = 0.55
axes[0].bar(frames, raw, width=1.0, color=['#D0D7DE' if not (46 <= i < 52) else colors['red'] for i in frames], edgecolor='none')
axes[0].axhline(raw.mean(), color='#6C757D', lw=0.8, ls='--')
axes[0].set_title('(a) Raw long window')
axes[0].text(4, .82, 'few abnormal frames\namid many normal frames', fontsize=7)
resid = np.random.default_rng(1).normal(0.12, 0.025, 100)
resid[46:52] += np.array([0.28,0.42,0.62,0.51,0.36,0.18])
axes[1].bar(frames, resid, width=1.0, color=[colors['teal'] if not (46 <= i < 52) else colors['red'] for i in frames], edgecolor='none')
axes[1].axhline(np.percentile(resid, 90), color='#6C757D', lw=0.8, ls='--')
axes[1].set_title('(b) Same-ID residual features')
axes[1].text(4, .82, 'local timing/payload\nchanges become visible', fontsize=7)
fig.suptitle('Why feature-before-window extraction matters', x=0.12, ha='left', fontsize=10, fontweight='bold')
save(fig, 'fig2_mechanism')

# Data from extra experiments
settings = ['Test01','Test02','Test03','Test04']
rawgb = [0.9677074993,0.9443764706,0.1344099379,0.2332780542]
graingb = [0.9423339710,0.9568167232,0.7958316633,0.7619047619]
gb_sample = [0.9163214582,0.9636322566,0.6712121212,0.5475504323]
rawtrans = [0.9641873278,0.1600389426,0.0252454418,0.0223889954]

# Figure 3: same-classifier representation control
fig, ax = plt.subplots(figsize=(6.5, 2.8))
x = np.arange(len(settings))
width = 0.35
ax.bar(x-width/2, rawgb, width, label='Raw-window + GB', color=colors['gray'])
ax.bar(x+width/2, graingb, width, label='GRAIN features + GB', color=colors['teal'])
ax.set_xticks(x, settings)
ax.set_ylabel('Attack-positive F1')
ax.set_ylim(0, 1.05)
ax.grid(axis='y', alpha=.25)
ax.legend(frameon=False, ncol=1, loc='upper right')
for i, (r,g) in enumerate(zip(rawgb, graingb)):
    delta = g-r
    ax.text(i, max(r,g)+0.035, f'{delta:+.02f}', ha='center', fontsize=7)
# Title is provided by the caption in the paper
save(fig, 'fig3_same_classifier')

# Figure 4: 2x2 delta heatmap for GRAIN+GB - Raw-window+GB
# matrix: rows vehicle known/unknown; cols attack known/unknown
mat = np.array([[graingb[0]-rawgb[0], graingb[2]-rawgb[2]], [graingb[1]-rawgb[1], graingb[3]-rawgb[3]]])
fig, ax = plt.subplots(figsize=(4.2, 3.0))
im = ax.imshow(mat, cmap='RdYlGn', vmin=-0.1, vmax=0.7)
ax.set_xticks([0,1], ['Known attack','Unknown attack'])
ax.set_yticks([0,1], ['Known vehicle','Unknown vehicle'])
for i in range(2):
    for j in range(2):
        ax.text(j,i,f'{mat[i,j]:+.03f}', ha='center', va='center', fontsize=9, fontweight='bold')
ax.set_title('F1 gain of GRAIN features over raw-window features')
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Delta F1')
save(fig, 'fig4_shift_matrix')

# Figure 5: ablation Test04 selected
variants = ['Timing only','w/o CAN-ID','Full feature set','w/o payload delta','Payload only','w/o same-ID timing']
f1s = [0.6732081911,0.6158730159,0.2332780542,0.2332780542,0.1189685984,0.2332780542]
# sort descending
order = np.argsort(f1s)[::-1]
variants = [variants[i] for i in order]
f1s = [f1s[i] for i in order]
fig, ax = plt.subplots(figsize=(5.8, 2.8))
ypos = np.arange(len(variants))
ax.barh(ypos, f1s, color=[colors['teal'] if 'Timing' in v else colors['blue'] if 'CAN' in v else colors['gray'] for v in variants])
ax.set_yticks(ypos, variants)
ax.invert_yaxis()
ax.set_xlim(0, 0.75)
ax.set_xlabel('Attack-positive F1 on Test04')
ax.grid(axis='x', alpha=.25)
for yv, val in zip(ypos, f1s):
    ax.text(val+0.015, yv, f'{val:.3f}', va='center', fontsize=7)
ax.set_title('Feature ablation indicates timing residual dominance under joint shift')
save(fig, 'fig5_ablation')

# Figure 6: low FPR curve Test04
budgets = np.array([1e-4,5e-4,1e-3,5e-3,1e-2])
grain_recall = np.array([0.4764119601,0.7488372093,0.8053156146,0.8053156146,0.8431893688])
raw_recall = np.array([0.0,0.1063122924,0.1401993355,0.1401993355,0.1401993355])
fig, ax = plt.subplots(figsize=(5.4, 3.0))
ax.plot(budgets, grain_recall, marker='o', label='GRAIN + GB', color=colors['teal'], lw=1.6)
ax.plot(budgets, raw_recall, marker='s', label='Raw-window + GB', color=colors['gray'], lw=1.6)
ax.set_xscale('log')
ax.set_xticks(budgets, ['1e-4','5e-4','1e-3','5e-3','1e-2'])
ax.set_ylim(0, 1.0)
ax.set_xlabel('Allowed FPR budget')
ax.set_ylabel('Detection rate at budget')
ax.grid(axis='both', alpha=.25)
ax.legend(frameon=False, loc='upper left')
# Title is provided by the caption in the paper
save(fig, 'fig6_low_fpr')

# Figure 7: efficiency
pipelines = ['Raw-window + GB','GRAIN + GB']
throughput = [98269.19,96999.56]
latency = [24497.44/24073,24818.09/24073]  # ms per window end-to-end approx
fig, ax = plt.subplots(figsize=(5.4, 2.6))
y = np.arange(len(pipelines))
ax.barh(y, throughput, color=[colors['gray'], colors['teal']])
ax.set_yticks(y, pipelines)
ax.set_xlabel('Frames per second (CPU-only)')
ax.grid(axis='x', alpha=.25)
for yi, val in zip(y, throughput):
    ax.text(val+1200, yi, f'{val/1000:.1f}k', va='center', fontsize=7)
ax.set_xlim(0, 110000)
ax.set_title('Deployment cost is dominated by stream parsing and feature extraction')
save(fig, 'fig7_efficiency')
