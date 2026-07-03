# Low-FPR and event-level revision

- Added protocol columns: `threshold_type`, `event_boundary`, `false_alarm_per_100k`, and `detection_delay`.
- Best-test rows are explicitly described as diagnostic score-separability evidence.
- Approximate event recall is explicitly described as non-official deployment evidence.
- Rows with high event recall but weak AUPR/low-FPR recall are described as insufficient for deployment claims.

Rows written: 8
