# Prediction/Table Consistency Audit

This audit checks whether existing per-sample prediction dumps reproduce the aggregate result tables when applying the recorded source threshold. Large differences mean the dump and aggregate row should not be treated as the same exact checkpoint/evaluation artifact.

- Prediction files checked: 15
- Rows with |aggregate F1 - source-threshold F1| > 0.02: 3

## Largest Differences

- ctt_test01 / concat_fusion: aggregate F1=0.8152, prediction@source-threshold F1=0.9472, diff=0.1320
- ctt_test01 / cmf_can: aggregate F1=0.9688, prediction@source-threshold F1=0.8649, diff=0.1038
- ctt_test01 / transformer: aggregate F1=0.9641, prediction@source-threshold F1=0.8957, diff=0.0684

## Interpretation

Use per-sample dumps for ranking/threshold/failure diagnostics. Use aggregate tables for headline historical results unless a fresh evaluation dump is regenerated from the exact same checkpoint and threshold-selection path.
