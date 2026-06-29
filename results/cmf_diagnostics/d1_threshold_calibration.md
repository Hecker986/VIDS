# D1 Threshold and Calibration Diagnosis

## Hypothesis
CMF-CAN may have good score ranking but poor threshold transfer/calibration.

## Evidence
- ROAD CMF-CAN AUROC/AUPR: 0.9431/0.8060; Transformer: 0.9335/0.7794.
- ROAD best-test-threshold F1: CMF-CAN 0.8212, Transformer 0.8279.
- ROAD default-threshold F1: CMF-CAN 0.8236, Transformer 0.8200.

## Answers
1. ROAD best-threshold F1 should be compared in the CSV; if CMF-CAN remains lower, representation/ranking alone is not enough.
2. AUROC/AUPR advantage only partially transfers to thresholded F1; calibration and threshold selection matter.
3. Low-FPR advantages come from score ordering under constrained thresholds, especially where recall@FPR is high despite poor F1.
4. Early stopping should include AUPR/Recall@FPR if deployment low-FPR is the target.
