# Related Work and Reproducibility Notes

This note is for paper writing only. It does not report reproduced baseline numbers unless the corresponding experiment exists in this repository.

## MIDS / Bidirectional Mamba

The most directly relevant recent work found is **MIDS: Detecting Stealthy Masquerade and Tampering Attacks on CAN Bus via Bidirectional Mamba**:

- Source: https://arxiv.org/abs/2606.18599
- Scope: CAN IDS for stealthy masquerade and tampering attacks.
- Reported idea: dual streams for CAN IDs and payloads, fused through bidirectional selective state-space modeling.
- Reported evaluation: private Tesla Model 3 data plus public benchmarks including ROAD, CrySyS, OTIDS and CT&T under a unified protocol.

Reproducibility fit for this repository:

- High conceptual relevance, especially because CMF-CAN also models multiple CAN modalities.
- Direct numeric comparison is not currently valid because this repository does not implement MIDS and does not use the same unified 5-fold protocol.
- A fair reproduction would require implementing the bidirectional Mamba block, matching the window protocol, and rerunning ROAD/CT&T/CrySyS under the same split rules.
- Recommended paper wording: cite MIDS as a strong recent state-space CAN IDS and discuss it as future/parallel work unless it is actually reproduced.

## Bi-Mamba-Class Models

Bi-Mamba is best treated here as an architectural family: bidirectional state-space sequence modeling for traffic/CAN windows.

Reproducibility fit:

- Feasible as a future baseline by replacing the frame-level Transformer encoder with a bidirectional Mamba/SSM encoder.
- It would not automatically replace CMF-CAN's window-statistics and ID-context branches; a fair variant should compare frame-only Bi-Mamba, concat fusion, and CMF-style fusion.
- Expected risk: implementation and hyperparameter choices can strongly affect latency and low-FPR behavior, so any reported comparison must include inference cost and constrained-FPR metrics.

## TrafficFormer / Transformer-Class Traffic Models

No stable, clearly matching CAN IDS paper named exactly `TrafficFormer` was identified in the current quick literature check. Treat this as a broader Transformer-based traffic modeling category unless a specific paper is selected.

Reproducibility fit:

- The repository already includes a Transformer baseline on frame-level sequences.
- A stronger TrafficFormer-style baseline would need a precise architecture definition: tokenization, temporal attention span, positional encoding, packet/window aggregation, and thresholding policy.
- Until a specific implementation is reproduced, do not write that CMF-CAN outperforms TrafficFormer.

## Suggested Paper Wording

Safe:

- "Recent state-space and Transformer-style sequence models provide strong frame-level temporal baselines for CAN IDS."
- "A full reproduction of MIDS/Bi-Mamba-style baselines is left to future work because it requires protocol-aligned training and evaluation."
- "Our current evidence focuses on controlled comparisons among Transformer, Concat-Fusion and CMF-CAN under the same processed VIDS protocol."

Unsafe:

- "CMF-CAN outperforms MIDS/Bi-Mamba/TrafficFormer."
- "TrafficFormer was reproduced."
- "Mamba-style models are weaker than CMF-CAN."

