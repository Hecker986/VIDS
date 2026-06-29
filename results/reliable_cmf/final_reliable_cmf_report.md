# Final Reliable-CMF-CAN Report

## What was implemented
Reliable-CMF-CAN was already registered and smoke-tested; this run trained it on ROAD and CT&T test02 for 50 epochs.

## What was trained
- ROAD / reliable_cmf_can / seed 42 / 50 epochs
- CT&T test02 / reliable_cmf_can / seed 42 / 50 epochs

## What improved
- ROAD F1 improves under Recall@FPR threshold selection from 0.7877 to 0.8137, but not enough to beat Transformer.

## What did not improve
- ROAD Reliable Full does not beat Transformer or old CMF-CAN.
- CT&T test02 Reliable Full remains at F1 0.1600 and does not beat -Ctx/anomaly policy.

## Best model per setting
See `final_model_selection.csv`.

## Recommended paper claim
Do not claim Reliable-CMF-CAN is the new main model yet. Claim it is an implemented prototype whose first training results show the need for simpler context-controlled or anomaly-calibrated variants.

## Remaining gap to CCF A / Security Four
Need trained ablations, test04 results, multi-seed stability, event-level metrics, and a working normality branch integrated into the Reliable score.
