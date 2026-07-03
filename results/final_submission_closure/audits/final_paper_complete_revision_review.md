# Final Paper Complete Revision Review

## What changed in this pass

The paper now includes two additional evidence figures that are directly supported by existing CSV/SVG/PDF artifacts:

1. **Ranking inversion**: shows that weighted-F1 and attack-F1 can rank detectors differently.
2. **GRAIN granularity**: shows sample, short-window, aggregate-window, and old fixed-window behavior on CT&T test04.

The low-FPR/event evidence remains in the main table rather than a separate figure, because adding that figure pushed the manuscript to 17 pages. Keeping it as Table 5 preserves the evidence while meeting the 16-page requirement.

## Current manuscript structure

- Figures: 5
- Tables: 5
- Bibliography entries: 29
- Cited keys: 29
- Uncited bibliography entries: 0
- Missing bibliography entries: 0
- Revised PDF pages: 16

## Figure/table evidence check

All added figures correspond to existing generated evidence:

- `figure4_ranking_inversion_scatter.pdf`
- `figure5_grain_granularity.pdf`

The manuscript does not include unrelated external-sanity figures in the main body. External data remain framed as sanity checks and limitations.

## Claim safety check

The revised PDF text was scanned for residual internal-baseline and unsafe-claim markers:

- No `CMF`
- No `Reliable`
- No `Concat`
- No `CAN-Tr`
- No `N/A`
- No claim that unknown attacks are solved
- No claim that public 0.998 is attack-F1
- No claim that CT&T or the original paper is wrong

## Submission judgment

The paper is now stronger than the previous 15-page version because it uses the remaining page to add two high-value visual arguments rather than padding. It is still not a "solved unknown attack" paper. The defensible submission story is:

**attack-centric evaluation + metric forensics + corrected benchmark + feature-preserving GRAIN-CAN baseline.**

This is the correct framing for CIVS / CCF B style review. For Security Four / CCF A, the remaining weaknesses are still external author confusion matrices/code, official event boundaries, and exact v1.5 alignment.
