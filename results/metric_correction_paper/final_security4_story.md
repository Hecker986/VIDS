# Final Security4 Story

1. Main line: metric/protocol correction for CT&T test04.
2. Original Table high F1 is potentially misleading because accuracy equals recall for many rows, consistent with weighted recall, and trivial all-normal reaches weighted/accuracy-like 0.998 while attack-F1 is zero.
3. Corrected attack-centric benchmark shows test04 remains difficult.
4. Best corrected row: `GradientBoosting / GRAIN_window_100` with attack-F1=0.6678.
5. Safe-CAN/GRAIN-CAN provide strong corrected baselines, not a solved unknown-attack claim.
6. This can support a CCF A/Security Four-style benchmark correction paper if framed as metric forensics plus corrected evaluation, not ordinary model tuning.
7. Remaining gaps: original authors' confusion matrices, official parameter grid/code, exact v1.5 source and official event boundaries.
