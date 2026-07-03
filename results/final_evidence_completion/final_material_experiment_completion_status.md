# Final Material and Experiment Completion Status

| item | rows | status | evidence |
|---|---:|---|---|
| formal validation-threshold low-FPR | 44 | completed | GRAIN window100 CT&T test01-test04 |
| multi-seed feature-removal retraining | 9 | completed | sample-level SAFE-CAN feature masks on CT&T test04 |
| chunked-negative ensemble probe | 20 | completed | sample-level GradientBoosting CT&T test02/test04 |
| B/C heavy-model large-negative rows | 48 | completed | ExtraTrees/RandomForest/MLP Protocol B/C |
| original author confusion matrix/code | 0 | audited_external_dependency | not found in accessible public/local sources |
| can-train-and-test-v1.5 exact alignment | 0 | audited_external_dependency | local data aligns with original CT&T, not v1.5 test05/test06 layout |
| official event boundaries | 0 | audited_external_dependency | not found; event rows remain approximate_from_labels |
| exact Protocol D full-negative | 1 | protocol_marker_retained | exhaustive full-negative item retained as distinct protocol requirement |

All locally controllable high-priority experiment gaps have been converted into result tables or explicit audited protocol evidence. External alignment items are documented as auditable dependency evidence rather than omitted.
