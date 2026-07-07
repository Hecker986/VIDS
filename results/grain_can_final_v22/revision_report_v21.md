# GRAIN-CAN v21 Revision Report

This revision addresses duplicated metric explanations around result tables.

## Changes

1. Kept the full metric definitions only in Section 5.3 (`Metrics and Operating Points`).
2. Established a single notation mapping in Section 5.3:
   - Precision = P
   - Recall = R
   - Attack-F1 = F_1^{atk}
   - FNR = false-negative rate
   - FPR = false-positive rate
   - AUPR and Recall@FPR = supplementary score-based metrics
3. Simplified Table 5 and Table 6 captions so they refer back to Section 5.3 instead of re-explaining every metric.
4. Kept table headers concise while preserving notation consistency.
5. Recompiled the paper and rendered pages around Section 5.3 / Tables 5-6 for layout verification.

## Output

- `grain_can_framework_style_v21.tex`
- `grain_can_framework_style_v21.pdf`
- `revision_report_v21.md`
