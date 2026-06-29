# Diagnostics Missing or Limited Items

The diagnostics intentionally avoid fabricated data. Items below were not generated as full new experiments because their required inputs or retraining artifacts are not currently available.

## D2 Modality Matrix

Some pairwise modality variants are not trained or implemented as saved artifacts:

- Context-only
- Frame+Stats
- Frame+Context
- Stats+Context

They are retained as missing rows in `results/cmf_diagnostics/tables/d2_modality_matrix.csv`.

## D7 Window Size Sensitivity

Only the current default window configuration is available. A true D7 experiment requires rebuilding processed windows/features for `window_size=50`, `100`, and `200`, then retraining or evaluating the selected models. No synthetic window-size result was created.

## D8 Objective and Early Stopping

This diagnostic consolidates existing objective/selection evidence rather than launching a new multi-seed loss and early-stopping sweep. A full D8 requires controlled retraining under weighted CE, focal loss, val F1 selection, val AUPR selection, and Recall@FPR selection.

## D9 Minimal Improvements

D9 uses existing real robust/SupCon/minimal-variant rows where available. New segment/top-k pooling, context masking, and gate entropy regularization were not implemented in this run because they require additional model changes and retraining. They remain recommended follow-up actions, not reported results.

## Practical Next Step

The highest-value next controlled experiment is a small multi-seed CT&T test02/test04 run comparing Full CMF-CAN, `-Ctx`, Stats+Frame, and calibrated-threshold selection. This directly targets the root causes ranked highest by the diagnostics: calibration mismatch, over-fusion, and ID-context shift.
