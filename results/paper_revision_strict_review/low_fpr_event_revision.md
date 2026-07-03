# Low-FPR and event-level revision

- Removed all-empty `false_alarm_per_100k` and `detection_delay` columns from the main paper table. These fields are discussed in text as unavailable without timestamp-calibrated event data for all listed rows.
- Kept protocol information in text: best-test rows are diagnostic score-separability evidence, and approximate event recall is not official deployment evidence.
- Best-test rows are explicitly described as diagnostic score-separability evidence.
- Approximate event recall is explicitly described as non-official deployment evidence.
- Rows with high event recall but weak AUPR/low-FPR recall are described as insufficient for deployment claims.
- Removed internal exploratory models from the main low-FPR comparison. The table now uses GRAIN, safe-feature GradientBoosting, public/default HistGradientBoosting protocols, old window100 Transformer, and all-normal.

Rows written: 6
