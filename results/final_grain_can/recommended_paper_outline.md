# Recommended Paper Outline Revision

1. Motivation: fixed long-window deep CAN IDS can destroy shifted-setting signal.
2. Measurement: compare official sample-level, short-window, feature-preserving aggregate-window, and old window=100 deep protocols.
3. Method: GRAIN-CAN as a feature-preserving granularity-aware evaluation and detector pipeline.
4. Stability: capped, 2x negative-cap, and 5x negative-cap evidence, with full-negative listed as a limitation.
5. Deployment: low-FPR upper-bound curves and approximate event-level analysis, clearly caveated.
6. Audit: past-only feature construction and leakage/protocol risk review.
7. Limits: test04 unknown vehicle + unknown attack is not solved; formal validation-threshold low-FPR and official event boundaries remain future work.
