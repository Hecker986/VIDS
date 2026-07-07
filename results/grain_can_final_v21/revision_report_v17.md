# GRAIN-CAN v17 Revision Report

## Scope
This revision focuses on rewriting the Related Work section in a more concrete security/IDS-paper style.

## Main changes
1. Replaced the previous broad, category-level Related Work paragraphs with category-organized but paper-specific discussion.
2. Added explicit descriptions of representative works in each category:
   - rule/timing/statistical IDS: Song et al., Taylor et al., Blevins et al., ECU fingerprinting, Viden;
   - learning/historical IDS: CANet, CAN-BERT, IDS-DEC, attention/Transformer/state-space designs, HistCAN;
   - signal-level/multimodal/explainable IDS: CANShield, X-CANIDS, CANival, signal-relation graph-transformer work;
   - graph/federated/deployment IDS: StatGraph, DGIDS, federated IDS, FPGA/hardware-counter/deterministic monitoring;
   - dataset and metric comparability: ROAD, CT&T, can-sleuth, ROAD comparative studies, surveys, ROC/PR and fixed-operating-point analysis.
3. For each category, clarified what the representative works do and how GRAIN-CAN differs.
4. Explicitly avoided claiming that GRAIN-CAN is stronger than signal-level, graph-based, or self-supervised methods. The text positions GRAIN-CAN as a lightweight residual representation rather than a general zero-day solution.
5. Kept the paper at 16 pages by compacting the bibliography while preserving all citations.

## Verification
- Recompiled with LaTeX.
- Final PDF page count: 16.
- No undefined citations or undefined references in the final compile log.
