# Low-FPR Objective Findings

- ROAD: Recall@FPR<=1e-3 selection raises thresholded F1 from 0.7877 to 0.8137, but it still does not beat Transformer F1 0.8279.
- CT&T test02: Recall@FPR<=1e-3 remains 0.4540 for Reliable-CMF-CAN, below old CMF-CAN/anomaly-policy evidence.
- Main metric should include Recall@FPR/AUPR, but changing threshold alone does not make Reliable-CMF-CAN the new main model.
