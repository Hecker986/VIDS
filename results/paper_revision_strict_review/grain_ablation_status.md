# GRAIN mechanism status

Full CT&T test04 feature-removal retraining evidence is now included from `results/final_evidence_completion/tables/grain_full_retraining_ablation.csv`.

Rows written: 9

Key interpretation:
- Removing `delta_t_same_id` collapses attack-F1, confirming same-ID timing as the dominant sample-level causal signal.
- `only_timing` is the strongest retrained feature subset in this sample-level feature-removal experiment.
- This table is a stricter feature-removal retraining study, not merely proxy feature importance.
