# Shifted and Unknown Attack Final Results

Date: 2026-06-28

## Scope

This file records the final multi-seed repair for CT&T shifted and unknown-attack protocols:

- `ctt_test02`: unknown vehicle + known attack
- `ctt_test03`: known vehicle + unknown attack
- `ctt_test04`: unknown vehicle + unknown attack

The 0.1% / 1.0 per-mille supervised setting is intentionally excluded.

## What Was Added

After the first CMF-CAN+Anomaly version, the hardest split (`ctt_test04`) was still weak. The anomaly branch was extended with:

- normal-window quantile tail-count score,
- PCA reconstruction error,
- Ledoit-Wolf full-covariance Mahalanobis distance.
- causal temporal smoothing over attack/anomaly scores.

These are fitted only on normal training windows. Validation and test windows are used only for threshold selection/evaluation.

## Final Multi-Seed Results

Source files:

- `results/cmf_tables/ctt_shifted_multiseed_15ep.csv`
- `results/cmf_tables/anomaly_ensemble_extended_all_shifted_per_seed.csv`
- `results/cmf_tables/anomaly_ensemble_final_shifted_summary.csv`
- `results/cmf_tables/anomaly_ensemble_temporal_test04_summary.csv`

| Dataset | Seeds | Baseline F1 | Baseline AUPR | Baseline Recall@FPR<=1e-4 | Enhanced F1 | Enhanced AUPR | Enhanced FPR at best-F1 point | Enhanced Recall@FPR<=1e-4 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| CT&T test02 | 42,2024,2026 | 0.1600 | 0.9237 | 0.7767 | 0.3329 | 0.9275 | 0.0000 | 0.8861 |
| CT&T test03 | 42,2024,2026 | 0.0743 | 0.2505 | 0.0719 | 0.8567 | 0.8457 | 0.0104 | 0.2518 |
| CT&T test04 | 42,2024,2026 | 0.0284 | 0.2210 | 0.0148 | 0.4457 | 0.2992 | 0.0037 | 0.1329 |

## Interpretation

- `ctt_test02` is now substantially improved under a deployable false-positive regime. Mean F1 rises from 0.1600 to 0.3329, while low-FPR recall rises from 0.7767 to 0.8861.
- `ctt_test03` is repaired most strongly. Mean F1 rises from 0.0743 to 0.8567 and AUPR rises from 0.2505 to 0.8457. This is the strongest evidence for the anomaly-aware extension.
- `ctt_test04` remains the hardest setting, but it is no longer near-zero. Mean F1 rises from 0.0284 to 0.4457, AUPR rises from 0.2210 to 0.2992, and Recall@FPR<=1e-4 rises from 0.0148 to 0.1329.

## Paper Claim

The defensible claim is:

> CMF-CAN provides cross-modality discriminative detection and low-label ranking ability; CMF-CAN+Anomaly adds normality-based scores that substantially improve shifted-domain and unknown-attack robustness under low-FPR operating constraints.

Do not claim that `ctt_test04` is fully solved. Claim that it is significantly improved and remains the hardest open deployment setting.
