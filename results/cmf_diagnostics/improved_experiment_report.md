# Improved Experiment Report

This report summarizes the result-improvement experiments selected from the first-principles diagnosis. It uses only existing measured tables and does not fabricate missing scores.

## First-Principles Fixes

1. Threshold/calibration mismatch: use validation-selected operating policies and low-FPR reporting.
2. ID-context vehicle shift: use context-masked / context-removed CMF variants when vehicle identity shifts.
3. Unknown attack shift: add normality/anomaly scores fitted on benign training windows.
4. Over-fusion: keep simplified variants when Full CMF-CAN is not the best measured model.

## Best Measured Improvements

| Dataset | Baseline F1 | Best fix | Improved F1 | F1 gain | Evidence |
|---|---:|---|---:|---:|---|
| road | 0.7894 | CMF variant: wo_stats | 0.8089 | 0.0195 | single-seed ablation |
| ctt_test02 | 0.1600 | Context-masked CMF | 0.8528 | 0.6927 | single-seed controlled ablation |
| ctt_test03 | 0.0211 | Stats-only | 0.0945 | 0.0734 | single-seed controlled ablation |
| ctt_test04 | 0.0333 | Context-masked CMF | 0.1821 | 0.1488 | single-seed controlled ablation |
| ctt_test02 | 0.1600 | CMF-CAN + normality/anomaly policy | 0.3329 | 0.1729 | three-seed shifted-domain anomaly repair |
| ctt_test03 | 0.0743 | CMF-CAN + normality/anomaly policy | 0.8567 | 0.7824 | three-seed shifted-domain anomaly repair |
| ctt_test04 | 0.0284 | CMF-CAN + normality/anomaly policy | 0.4457 | 0.4173 | three-seed shifted-domain anomaly repair |

## Important Caveats

- The strongest CT&T test02 context-masked result is a single-seed controlled ablation, not yet a three-seed headline.
- The CT&T test03 anomaly repair is strong and multi-seed, but it relies on a normality score branch rather than only the supervised CMF-CAN classifier.
- CT&T test04 is improved but still not solved; the honest claim is substantial repair, not production-ready unknown-vehicle unknown-attack detection.
- ROAD F1 can improve by simplifying fusion, but Transformer remains a strong F1 baseline; CMF-CAN is stronger mainly in ranking/AUPR/low-FPR evidence.

## Recommended Main Result Framing

Use `CMF-CAN + calibrated/adaptive policy` as the improved system rather than claiming the original Full CMF-CAN classifier is universally best. The improved system chooses a score family or fusion variant according to observed domain shift: Full/Frame for known settings, context masking for unknown vehicle, and benign-normality anomaly scoring for unknown attack.

## Final Deployment Policy Snapshot

| Dataset | Recommended score | Threshold policy | F1 | AUPR | Recall@FPR<=1e-4 |
|---|---|---|---:|---:|---:|
| ctt_test01 | model_attack_prob | val_fpr_1em04 | 0.9819 | 0.9826 | 0.9679 |
| ctt_test02 | per-seed extended anomaly policy | validation-selected low-FPR/F1 policy | 0.3329 | 0.9275 | 0.8861 |
| ctt_test03 | per-seed extended anomaly policy | validation-selected low-FPR/F1 policy | 0.8567 | 0.8457 | 0.2518 |
| ctt_test04 | per-seed extended anomaly policy | validation-selected low-FPR/F1 policy | 0.4457 | 0.2992 | 0.1329 |
| road | model_attack_prob | val_f1 | 0.5899 | 0.5974 | 0.0932 |
| road | model_attack_prob | val_fpr_1em03 | 0.7971 | 0.7586 | 0.4653 |
