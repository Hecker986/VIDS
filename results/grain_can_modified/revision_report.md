# GRAIN-CAN manual revision report

## What I changed

1. Rewrote the paper around a single method claim: same-ID local behavior residuals before fixed-window aggregation.
2. Strengthened the method section with per-ID history state, feature groups, formulas, component table, and train/inference protocol.
3. Reframed GRAIN-CAN as a supervised feature-based CAN IDS pipeline rather than a neural model or evaluation protocol.
4. Moved away from audit-report style writing and made limitations appear in Discussion rather than throughout the abstract/method.
5. Rewrote the evaluation narrative around cross-shift stability, raw-window comparison, ablation, window sensitivity, and supplementary score metrics.
6. Adjusted the ablation interpretation: the current evidence supports same-ID timing residuals as the dominant signal; payload and ID groups should not be overclaimed.
7. Rebuilt all figures with a cleaner conference-style visual language and concise captions.
8. Kept score metrics such as AUPR and Recall@FPR as supplementary evidence only.

## Important remaining gaps

1. The paper still needs real efficiency measurements if it wants to claim lightweight deployment.
2. The method would be stronger with a same-classifier representation control: raw-window features + GB vs GRAIN features + GB.
3. The current ablation weakens broad multi-feature claims; timing residuals are the strongest supported mechanism.
4. Stronger comparison to recent self-supervised, signal-level, graph, and sequence IDS pipelines requires local reproduction under the same CT&T split and metric definitions.
5. Environment details, hyperparameters, and exact scripts should be filled from the repository logs rather than invented.

## Deliverables

- `grain_can_framework_style_v2.tex`
- `grain_can_framework_style_v2.pdf`
- `figures/*.pdf`
- `figures/*.svg`
