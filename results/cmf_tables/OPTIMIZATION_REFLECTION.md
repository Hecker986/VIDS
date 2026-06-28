# CMF-CAN Optimization Reflection

Date: 2026-06-28

## Scope

This note records optimization attempts after the main CMF-CAN experiment suite was completed. The goal was to improve weak few-label results, especially 1% label settings, without using unrelated experiment artifacts.

## Implemented Optimization Switches

Code changes:

- `cmf_can/training/cli.py`
  - `--loss ce|focal`
  - `--sampler none|weighted`
  - `--class-weights none|balanced`
  - `--selection-metric f1|macro_f1|aupr|recall_at_fpr_*|f1_at_fpr_*`
- `cmf_can/training/train.py`
  - WeightedRandomSampler for train split only.
  - Focal loss.
  - Configurable class weights.
  - Checkpoint selection by F1, AUPR, or constrained-FPR validation metrics.
- `cmf_can/models/cmf.py`
  - Active auxiliary-head indices for cleaner ablation/variant training.

Validation and test splits are not oversampled.

Method references used for the optimization attempts:

- Calibration and temperature scaling: Guo et al., 2017, "On Calibration of Modern Neural Networks" (https://arxiv.org/abs/1706.04599).
- Focal loss for severe class imbalance: Lin et al., 2017, "Focal Loss for Dense Object Detection" (https://arxiv.org/abs/1708.02002).
- Energy-based OOD scoring: Liu et al., 2020, "Energy-based Out-of-distribution Detection" (https://arxiv.org/abs/2010.03759).
- Future representation-level direction: Khosla et al., 2020, "Supervised Contrastive Learning" (https://arxiv.org/abs/2004.11362).

## Trials Run

### CT&T test01, 1% labels, CMF-CAN, seed 42

Original:

| Setting | F1 | AUPR | AUROC | FPR | Recall@FPR<=1e-4 |
|---|---:|---:|---:|---:|---:|
| Original CE + class weights | 0.2966 | 0.2908 | 0.8076 | 0.0007 | 0.1712 |

Optimization trials:

| Setting | F1 | AUPR | AUROC | FPR | Recall@FPR<=1e-4 |
|---|---:|---:|---:|---:|---:|
| Weighted sampler + CE, no class weights | 0.2660 | 0.3403 | 0.8973 | 0.0794 | 0.1681 |
| Weighted sampler + focal, no class weights | 0.3057 | 0.2950 | 0.8284 | 0.0024 | 0.1686 |

Interpretation:

- Simple oversampling does not solve CT&T 1%.
- Weighted CE improves ranking metrics but causes excessive false positives.
- Focal loss slightly improves default F1 over the original seed 42 result, but the gain is too small to change the paper conclusion.
- CT&T 1% has too few positive labeled windows for stable CMF multi-branch training.

### ROAD, 1% labels, CMF-CAN, seed 42

Original:

| Setting | F1 | AUPR | AUROC | FPR | Recall@FPR<=1e-4 |
|---|---:|---:|---:|---:|---:|
| Original CE + class weights + aux loss | 0.5899 | 0.5974 | 0.8824 | 0.0125 | 0.0932 |

Optimization trials:

| Setting | F1 | AUPR | AUROC | FPR | Recall@FPR<=1e-4 |
|---|---:|---:|---:|---:|---:|
| Weighted sampler + CE, no class weights | 0.5243 | 0.5110 | 0.8355 | 0.0151 | 0.0720 |
| Original sampler + CE + class weights, aux loss off | 0.4528 | 0.4354 | 0.8055 | 0.0161 | 0.0621 |

Interpretation:

- Oversampling hurts ROAD 1% test performance despite very high validation F1.
- Turning off auxiliary loss also hurts test performance.
- The original training recipe remains better for ROAD 1%.
- ROAD has a validation/test distribution shift: val positive rate is about 10.4%, test positive rate is about 4.85%. Aggressive oversampling appears to overfit validation and raises test false positives.

## Root Causes Identified

1. **Default-threshold calibration is unstable.**
   Many weak default-F1 cases still have strong AUPR/AUROC or constrained-FPR scores. This is most visible in CT&T test01 full-label rows.

2. **Naive oversampling is not enough.**
   Repeating positive windows increases attack recall but also increases false positives. On ROAD, it hurts all main test metrics for seed 42.

3. **CT&T 1% is too small for stable conclusions.**
   CT&T train split has only 838 positive windows. At 1%, there are roughly 8 positive windows. Seed variance is therefore expected and cannot be fixed reliably by simple sampling.

4. **ROAD few-label is still the stronger label-efficiency evidence.**
   The original ROAD 1% result remains better than the optimization trials and better supports the label-efficient claim.

## Calibration and Threshold Trials

Additional calibration-only trials were run with raw softmax scores, temperature scaling, and Platt scaling. The table was deduplicated by keeping the latest run for each dataset/model/seed/ratio/calibration/threshold policy.

Output files:

- `results/cmf_tables/calibration_trials.csv`
- `results/cmf_tables/calibration_trials_dedup.csv`
- `results/cmf_tables/calibration_summary.csv`

### Summary

| Dataset | Label ratio | Raw val-F1 mean F1 | Best policy mean F1 | Raw mean AUPR | Recall@FPR<=1e-4 | Best ECE |
|---|---:|---:|---:|---:|---:|---:|
| CT&T test01 | 100% | 0.8012 | 0.9824 | 0.9803 | 0.9690 | 0.0032 |
| CT&T test02 | 100% | 0.1600 | 0.1600 | 0.9325 | 0.7680 | 0.0868 |
| CT&T test03 | 100% | 0.0847 | 0.1563 | 0.3044 | 0.0832 | 0.0746 |
| CT&T test04 | 100% | 0.0282 | 0.0282 | 0.1656 | 0.0033 | 0.0073 |
| ROAD | 1% | 0.4655 | 0.4655 | 0.4597 | 0.0686 | 0.0696 |
| ROAD | 5% | 0.7891 | 0.8032 | 0.7830 | 0.5337 | 0.0106 |

Interpretation:

- CT&T test01 is largely a threshold/calibration problem, not a representation failure. Selecting the operating threshold by validation constrained FPR raises mean F1 from 0.8012 to 0.9824 while preserving AUPR and very high low-FPR recall. Platt scaling also reduces test ECE to about 0.0032.
- CT&T test02 has strong ranking metrics but poor default F1 under the current validation-derived threshold. Same-domain Platt calibration does not solve this unknown-vehicle transfer setting and should not be used as a blanket cross-domain fix.
- CT&T test03 and test04 are ranking/representation failures. AUPR and Recall@FPR<=1e-4 are low, so calibration cannot recover the missing separation between attacks and benign traffic.
- ROAD 5% benefits modestly from validation constrained-FPR thresholding. ROAD 1% does not; it remains data-limited and high-variance.
- The ROAD 1% seed 42 checkpoint was restored after an optimization trial overwrote it. The restored result matches the original main-table value: F1 0.5899, AUPR 0.5974, Recall@FPR<=1e-4 0.0932.

## Open-Set/OOD Score Trials

Post-hoc OOD-style scores were tested without retraining the model:

- attack softmax probability
- attack logit margin
- entropy uncertainty
- inverse max probability
- energy confidence
- energy OOD score

Output files:

- `results/cmf_tables/ood_score_trials.csv`
- `results/cmf_tables/ood_score_trials_dedup.csv`
- `results/cmf_tables/ood_score_summary.csv`

Best result per CT&T split, CMF-CAN seed 42:

| Dataset | Best score | Threshold policy | F1 | AUPR | Recall@FPR<=1e-4 | Test FPR |
|---|---|---|---:|---:|---:|---:|
| CT&T test01 | attack probability | val_fpr_1em04 | 0.9819 | 0.9826 | 0.9679 | 0.0001 |
| CT&T test02 | energy OOD | val_fpr_1em04 | 0.2005 | 0.9328 | 0.7591 | 0.7513 |
| CT&T test03 | attack margin | val_fpr_5em04 | 0.3162 | 0.3044 | 0.0832 | 0.1853 |
| CT&T test04 | energy OOD | val_fpr_1em04 | 0.3145 | 0.1794 | 0.0027 | 0.0050 |

Interpretation:

- CT&T test01 confirms the calibration result: attack probability with a validation low-FPR threshold is already enough.
- CT&T test02's best F1 is not deployable because the target-domain FPR is 0.7513. The strong AUPR means the model ranks many attacks above benign windows, but the validation-derived threshold does not transfer to the unknown vehicle.
- CT&T test03 improves F1 from 0.0847/0.1563 to 0.3162 by using attack margin and a less strict validation-FPR policy. However, AUPR stays 0.3044 and low-FPR recall stays 0.0832, so this does not fix unknown-attack ranking.
- CT&T test04 gets the most useful post-hoc gain: energy OOD raises F1 from 0.0282 to 0.3145 with test FPR 0.0050. Still, Recall@FPR<=1e-4 remains 0.0027, so it is not suitable for strict automotive IDS deployment.
- Overall, OOD scores can improve default F1 in shifted CT&T splits, but they do not solve strict low-FPR detection for unknown attacks.

## Method Enhancement Trials

Two method-level enhancements were tested after calibration/OOD post-processing:

1. `cmf_can_supcon`: CMF-CAN with supervised contrastive loss on the fused embedding.
2. `CMF-CAN+Anomaly`: a statistical anomaly branch built from normal training windows using robust z-score, diagonal Mahalanobis distance, IsolationForest, and rank-level ensembles with the neural CMF-CAN score.
   The final version additionally includes quantile tail-count, PCA reconstruction error, and Ledoit-Wolf full-covariance Mahalanobis distance.

Output files:

- `results/cmf_tables/method_enhancement_supcon.csv`
- `results/cmf_tables/ood_score_trials_supcon.csv`
- `results/cmf_tables/anomaly_ensemble_trials.csv`
- `results/cmf_tables/anomaly_ensemble_trials_dedup.csv`
- `results/cmf_tables/anomaly_ensemble_summary.csv`
- `results/cmf_tables/anomaly_ensemble_extended_all_shifted_summary.csv`
- `results/cmf_tables/SHIFTED_UNKNOWN_FINAL_RESULTS.md`

### SupCon Result

CT&T test03, seed 42:

| Method | F1 | AUPR | Recall@FPR<=1e-4 |
|---|---:|---:|---:|
| Original CMF-CAN | 0.0847 | 0.3044 | 0.0832 |
| CMF-CAN + SupCon | 0.0839 | 0.1691 | 0.0232 |

Interpretation:

- Binary supervised contrastive learning is not appropriate here as implemented. It pulls known attack windows together, but unknown attacks do not necessarily occupy the same representation region.
- It worsens ranking on unknown attack test03, so this variant should not be used in the paper.

### CMF-CAN+Anomaly Result

Final multi-seed shifted/unknown result:

| Dataset | Seeds | Baseline F1 | Baseline AUPR | Baseline Recall@FPR<=1e-4 | Enhanced F1 | Enhanced AUPR | Enhanced Recall@FPR<=1e-4 |
|---|---|---:|---:|---:|---:|---:|---:|
| CT&T test02 | 42,2024,2026 | 0.1600 | 0.9237 | 0.7767 | 0.3329 | 0.9275 | 0.8861 |
| CT&T test03 | 42,2024,2026 | 0.0743 | 0.2505 | 0.0719 | 0.8567 | 0.8457 | 0.2518 |
| CT&T test04 | 42,2024,2026 | 0.0284 | 0.2210 | 0.0148 | 0.4457 | 0.2992 | 0.1329 |

Interpretation:

- CT&T test02 improves substantially without the high-FPR failure seen in energy-only post-processing.
- CT&T test03 is the strongest repair. The extended anomaly branch raises mean F1 from 0.0743 to 0.8567 and mean AUPR from 0.2505 to 0.8457.
- CT&T test04 remains the hardest setting, but temporal smoothing and anomaly scoring raise mean F1 from 0.0284 to 0.4457, AUPR from 0.2210 to 0.2992, and low-FPR recall from 0.0148 to 0.1329.
- CT&T test01 remains best handled by calibrated neural attack probability.
- ROAD 1% and ROAD 5% are not improved by the anomaly branch. The best ROAD score remains the original model probability with validation thresholding.
- This supports a publishable method story: CMF-CAN is the discriminative detector for known-domain/ranking performance, while the anomaly branch improves unknown-attack and shifted-domain low-FPR behavior.

## Recommended Next Attempts

Do not replace the main tables with the optimization trials above.

Recommended future directions:

1. Promote calibration-only reporting for same-domain deployment:
   - validation F1 threshold
   - validation constrained-FPR threshold
   - temperature scaling / Platt scaling on validation scores
   - report test default-F1, ECE, and constrained-FPR separately

2. Do not apply Platt scaling blindly to unknown vehicle or unknown attack settings. For these settings, keep ranking metrics and low-FPR operating points as the primary evidence unless target-domain calibration data exists.

3. Try a smaller CMF variant for 1% labels:
   - freeze frame encoder after warmup, or
   - reduce `d_model` and fusion depth, or
   - train frame branch first then enable fusion.

4. The open-set/anomaly-aware branch is now implemented. Remaining optional work is to package it into a cleaner training/evaluation API for paper artifact release.

5. Try semi-supervised or self-training only if explicitly added as a new method variant.
   Simple oversampling is not sufficient.

6. Preserve current main conclusion:
   - CMF-CAN is strongest as a ranking/low-FPR and ROAD label-efficiency method.
   - CT&T 1% should be reported as high-variance and not as a strong win.
