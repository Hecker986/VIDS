# Final Metric Forensics Verdict

1. Table 13 reported F1 most likely cannot be treated as attack-positive F1 without additional proof. For supervised GradientBoosting-style rows, task-stated 0.998-level F1 is much closer to weighted-F1 / accuracy / normal-F1 than to attack-positive F1.
2. Current evidence does not support that public 0.998 is attack-positive test04 detection on the aligned original set_01. The best available sample-level corrected attack-F1 is 0.5862, not 0.998.
3. Public 0.998 should not be used as the true unknown-attack detection target until metric definition, confusion matrices and v1.5/protocol alignment are proven.
4. Class imbalance alone is sufficient to produce high-looking metrics: predict-all-normal gives accuracy=0.998923, normal-F1=0.999461, weighted-F1=0.998384, but attack-F1=0.000000.
5. Close 0.998 matching hypotheses observed in local rows: {'weighted_f1': 7, 'normal_f1': 2, 'attack_f1': 2}. BIRCH specifically is not explained by deployable validation mapping; its unsafe test-majority mapping cannot be verified because cluster assignments were not saved.
6. Existing Safe-CAN 0.586 attack-positive F1 is the strongest available sample-level corrected row from public reproduction; corrected benchmark best row is `GradientBoosting / window_100` with F1=0.6678.
7. The project should pivot to a benchmark metric/protocol correction paper unless exact Table 13 confusion matrices prove otherwise.
