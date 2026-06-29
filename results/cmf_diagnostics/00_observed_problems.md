# 00 Observed Problems

This file summarizes observed problems before new diagnosis. It uses only existing result files.

## Required observations
1. ROAD: CMF-CAN has better AUROC/AUPR than Transformer, but lower thresholded F1/Macro-F1.
2. Full CMF-CAN is weaker than simplified variants in several ablations; simplified models must be kept.
3. CT&T unknown settings remain weak, especially unknown attack and unknown vehicle + unknown attack.
4. Few-label results are unstable across ratios and datasets.
5. Low-FPR behavior has a real bright spot in CT&T test02, but not uniformly across all shifts.
6. Per-attack results show hard attack types, including fuzzing/interval/systematic and some malfunction-like low-recall cases.

## Files read
- `results/cmf_tables/paper_table_overall_main_results_refined.csv` exists=True
- `results/cmf_tables/paper_table_ablation_refined.csv` exists=True
- `results/cmf_tables/paper_table_few_label_refined.csv` exists=True
- `results/cmf_tables/paper_table_ctt_generalization_refined.csv` exists=True
- `results/cmf_tables/paper_table_low_fpr_refined.csv` exists=True
- `results/cmf_tables/paper_table_gate_weights.csv` exists=True
- `results/cmf_tables/paper_table_per_attack_results.csv` exists=True
- `results/cmf_tables/paper_readiness_review.md` exists=True
- `results/cmf_tables/missing_inputs_report.md` exists=True
