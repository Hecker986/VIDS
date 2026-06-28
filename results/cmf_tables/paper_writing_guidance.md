# Paper Writing Guidance

## Can Write
- CMF-CAN provides a structured way to integrate frame-level sequence, window-level statistics and ID-context features.
- On ROAD, CMF-CAN improves ranking-oriented AUROC/AUPR over Transformer, although thresholded F1 is lower.
- On CT&T known vehicle + known attack, CMF-CAN is competitive and slightly stronger in F1/Macro-F1.
- In CT&T unknown vehicle + known attack low-FPR analysis, CMF-CAN has a clear recall advantage over Transformer and Concat-Fusion.
- CT&T unknown-setting ablation shows window-level statistics are a robust shifted-setting signal; removing window statistics usually hurts low-FPR behavior.
- Unknown-setting ablation also shows Full CMF-CAN is not always the best deployable variant, so the contribution should be framed as a system study of cross-modality fusion rather than a universal model win.

## Cannot Write
- Do not claim CMF-CAN consistently outperforms Transformer.
- Do not claim CMF-CAN solves unknown attack or unknown vehicle + unknown attack generalization.
- Do not claim Full CMF-CAN is always the best ablation variant.
- Do not claim ID context is always beneficial; unknown vehicle settings can favor variants without ID context.
- Do not present HCRL/Car-Hacking as hard evidence; they are sanity checks.

## Recommended Positioning
Best fit: Cross-modality feature fusion for CAN IDS, with deployment-oriented low-FPR operating analysis. Label-efficient CAN IDS can be a supporting angle, not the headline.

## Strict Claim Guardrails
- Prefer: 'CMF-CAN and its ablations reveal which modalities transfer under each CT&T shift.'
- Prefer: 'Window statistics are the most stable shifted-setting signal in our CT&T unknown ablation.'
- Avoid: 'CMF-CAN consistently solves generalization.'
- Avoid: 'Cross-modal attention/gating is always beneficial.'

## Risks
- ROAD F1/Macro-F1 are worse than Transformer.
- Few-label results are mixed across ROAD and CT&T.
- Full CMF-CAN is not always best in ablation.
- CT&T test03/test04 have low absolute F1.
- HCRL/Car-Hacking are too easy and should be appendix sanity checks.

## Recommended Additional Experiments
P0: complete for this package: CT&T prediction/gate dumps, unknown-setting ablation, shifted PR/ROC inputs, and low-FPR recomputation from scores.
P1: remaining optional work: more seeds for unknown ablation and external industrial baselines.
P2: longer-term work: more external datasets and full reproduction of recent sequence/state-space baselines.
