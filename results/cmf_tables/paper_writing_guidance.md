# Paper Writing Guidance

## Can Write
- CMF-CAN provides a structured way to integrate frame-level sequence, window-level statistics and ID-context features.
- On ROAD, CMF-CAN improves ranking-oriented AUROC/AUPR over Transformer, although thresholded F1 is lower.
- On CT&T known vehicle + known attack, CMF-CAN is competitive and slightly stronger in F1/Macro-F1.
- In CT&T unknown vehicle + known attack low-FPR analysis, CMF-CAN has a clear recall advantage over Transformer and Concat-Fusion.

## Cannot Write
- Do not claim CMF-CAN consistently outperforms Transformer.
- Do not claim CMF-CAN solves unknown attack or unknown vehicle + unknown attack generalization.
- Do not claim Full CMF-CAN is always the best ablation variant.
- Do not present HCRL/Car-Hacking as hard evidence; they are sanity checks.

## Recommended Positioning
Best fit: Cross-modality feature fusion for CAN IDS, with deployment-oriented low-FPR operating analysis. Label-efficient CAN IDS can be a supporting angle, not the headline.

## Risks
- ROAD F1/Macro-F1 are worse than Transformer.
- Few-label results are mixed across ROAD and CT&T.
- Full CMF-CAN is not always best in ablation.
- CT&T test03/test04 have low absolute F1.
- HCRL/Car-Hacking are too easy and should be appendix sanity checks.

## Recommended Additional Experiments
P0: complete CT&T prediction/gate dumps, CT&T unknown-setting ablations, and low-FPR recomputation from scores.
P1: PR/ROC for ROAD and shifted CT&T, per-attack analysis, failure cases, calibration bins.
P2: embeddings for UMAP/t-SNE, more external datasets, additional industrial baselines.
