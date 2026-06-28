# Experiment Gap Fix Plan

Date: 2026-06-28

## Corrected Coverage

The project has already run 1%, 5%, 10%, 20%, and 100% label-ratio experiments for:

- ROAD: `cmf_can`, `concat_fusion`, `transformer`, 3 seeds.
- CT&T test01: `cmf_can`, `concat_fusion`, `transformer`, 3 seeds.

Coverage summary:

- `results/cmf_tables/label_ratio_coverage_summary.csv`
- `results/cmf_tables/LABEL_RATIO_COVERAGE_REPORT.md`

Therefore, 10% and 20% are not missing. They were missing only from the short deployment-policy table.

## Metric Repair

Do not remove default F1. It remains necessary for comparability.

However, for automotive IDS deployment, default F1 must not be the only headline metric. The main paper tables should include:

- F1 at validation-selected threshold.
- AUPR.
- AUROC.
- Recall@FPR<=1e-4.
- F1@FPR<=1e-4 or F1@FPR<=1e-3.
- ECE for calibrated scores.

Rationale:

- Default F1 is threshold-sensitive and can be misleading under vehicle/domain shift.
- AUPR captures ranking quality under class imbalance.
- Recall@FPR<=1e-4 reflects deployability.
- ECE and calibration show whether scores can be used operationally.

## Results That Are Already Defensible

### ROAD

- At 1%, CMF-CAN is the best model by F1, AUROC, AUPR, and low-FPR recall.
- At 5%, 10%, 20%, and 100%, Transformer often wins default F1, but CMF-CAN wins AUPR and/or low-FPR behavior.
- Paper claim should be: label efficiency at 1%, ranking quality, and low-FPR operation.
- Paper should not claim universal default-F1 superiority.

### CT&T test01

- 1%, 5%, 10%, 20%, and 100% are already covered.
- CMF-CAN is strong but not always the best default-F1 model.
- At 100%, calibrated CMF-CAN is excellent for AUPR and low-FPR recall.
- Paper claim should be: same-domain CMF-CAN scores are highly rankable and calibratable, not always default-F1 dominant.

### CT&T test03/test04

- CMF-CAN+Anomaly is a real method enhancement.
- test03 improves from AUPR 0.3044 to 0.8145 and Recall@FPR<=1e-4 from 0.0832 to 0.2233.
- test04 improves Recall@FPR<=1e-4 from 0.0033 to 0.1136, but remains the hardest setting.

## Experiments Completed After This Plan

Priority 1 was completed:

- CMF-CAN checkpoints were trained for CT&T test02/test03/test04 with seeds 2024 and 2026.
- CMF-CAN+Anomaly was rerun on CT&T test02/test03/test04 with seeds 42, 2024, and 2026.
- The anomaly branch was improved with quantile tail-count, PCA reconstruction error, and Ledoit-Wolf Mahalanobis scores.
- Final results are in `results/cmf_tables/SHIFTED_UNKNOWN_FINAL_RESULTS.md`.

Final shifted/unknown multi-seed summary:

| Dataset | Baseline F1 | Baseline AUPR | Baseline Recall@FPR<=1e-4 | Enhanced F1 | Enhanced AUPR | Enhanced Recall@FPR<=1e-4 |
|---|---:|---:|---:|---:|---:|---:|
| CT&T test02 | 0.1600 | 0.9237 | 0.7767 | 0.3329 | 0.9275 | 0.8861 |
| CT&T test03 | 0.0743 | 0.2505 | 0.0719 | 0.8567 | 0.8457 | 0.2518 |
| CT&T test04 | 0.0284 | 0.2210 | 0.0148 | 0.4457 | 0.2992 | 0.1329 |

## Experiments Still Worth Running

Priority 2:

- Add final paper tables that combine:
  - default F1,
  - calibrated F1,
  - AUPR,
  - Recall@FPR<=1e-4,
  - ECE.
- This fixes the metric presentation problem without hiding weak default-F1 rows.

Priority 3:

- ROAD anomaly branch at 10%, 20%, and 100% is optional.
- It already failed to improve ROAD 1% and 5%, so this is not essential for the main claim.

Priority 4:

- 0.1% / 1.0 per mille should not be run as pure supervised classification.
- If required, redesign it as benign-only/self-supervised anomaly learning with only per-mille malicious labels used for threshold selection.

## Recommended Paper Claim After Fixes

CMF-CAN is a cross-modality CAN IDS that improves low-label ranking and low-FPR detection. CMF-CAN+Anomaly extends it to shifted and unknown-attack settings. The method is strongest under 1% ROAD labels, CT&T same-domain calibration, and CT&T unknown-attack anomaly scoring.

Do not claim all splits are solved or that 0.1% supervised learning is already achieved.
