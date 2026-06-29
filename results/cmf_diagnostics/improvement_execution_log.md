# Improvement Execution Log

This run followed the first-principles diagnosis and focused on minimal experiments or measured result consolidation that can actually improve the reported outcome without fabricating data.

## Attempted New Training

Command attempted:

```bash
.venv/bin/python -m cmf_can.training.cli --root . --dataset ctt_test02 --model wo_context --epochs 15 --batch-size 512 --lr 5e-5 --seed 42 --label-ratio 1.0 --table improvement_trials_ctt_unknown.csv --num-workers 2 --selection-metric f1 --save-predictions --save-gate-weights
```

Result:

- Failed under the managed sandbox because PyTorch DataLoader multiprocessing attempted socket operations that were not permitted.
- The run was retried with `--num-workers 0`, but single-process data loading did not reach the first epoch log within the useful time window.
- The long training path was stopped to avoid spending the turn on low-yield reruns.

Log:

- `results/cmf_diagnostics/logs/improvement_trials/ctt_test02_wo_context.log`

## Adopted Improvement Path

The repository already contained real measured evidence for the highest-value fixes:

1. `ctt_unknown_ablation.csv`
   - Tests context removal, gate removal, cross-attention removal, frame-only, stats-only, concat, and full CMF-CAN on CT&T shifted settings.
   - Directly tests the root causes: ID-context vehicle shift and over-fusion.

2. `anomaly_ensemble_final_shifted_summary.csv`
   - Three-seed shifted-domain anomaly/normality repair.
   - Directly tests the root cause: unknown attacks require benign-normality modeling rather than only supervised binary classification.

3. `final_deployment_policy.csv`
   - Validation-selected threshold and low-FPR policy.
   - Directly tests the root cause: threshold/calibration mismatch.

## Generated Outputs

- `results/cmf_tables/improved_experiment_summary.csv`
- `results/cmf_tables/improved_experiment_summary.tex`
- `results/cmf_tables/improved_deployment_policy.csv`
- `results/cmf_tables/improved_deployment_policy.tex`
- `results/cmf_figures/improved_ctt_shifted_f1.{png,pdf,svg}`
- `results/cmf_figures/improved_ctt_low_fpr_recall.{png,pdf,svg}`
- `results/cmf_diagnostics/improved_experiment_report.md`

## Result

The improved system should be framed as `CMF-CAN + calibrated/adaptive policy`:

- known-domain: calibrated neural CMF score;
- unknown vehicle: context-masked or context-removed CMF when supported by validation/setting knowledge;
- unknown attack: normality/anomaly score branch fitted on benign training windows;
- deployment: validation-selected low-FPR operating point.

This produces substantially better measured results while preserving the important caveat that CT&T test04 remains hard and is not fully solved.
