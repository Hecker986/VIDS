# Final layout and baseline cleanup report

## Changes made

- Rebuilt Figure 1 so the bottom metric pills no longer crowd together. The redundant bottom metric pills and the separate orange metric-trap box were removed because the same concepts are already listed in the metric-audit and corrected-evaluation boxes.
- Removed internal exploratory model labels from the main paper and the external-sanity figure path. The final PDF/TeX no longer contains CMF-CAN, Reliable-CMF-CAN, Concat-Fusion, CAN-Tr+, or TFS as paper baselines.
- Removed non-core Figure 7/8 style material from the main paper. External sanity and evidence-closure points remain as concise text limitations rather than extra figures.
- Removed duplicate Algorithm 1 and duplicate bibliography entries.
- Kept only core paper figures in the revised manuscript: pipeline, metric forensics, and corrected benchmark. Ranking, granularity, and low-FPR evidence are represented by corresponding main tables to keep the manuscript within the 16-page limit.
- Reduced figure width to 0.88 text width to keep figures readable while satisfying the page budget.

## Verification

- Compiled PDF: `results/paper_revision_strict_review/attack_centric_can_ids_paper_revised.pdf`
- Page count: 16 pages under the local fallback article class.
- Figure 1 visual check: no bottom-box crowding or text-over-box overlap remains.
- Residual check: no `CMF`, `Reliable`, `Concat`, `CAN-Tr`, `CAN-Transformer`, `TFS`, `N/A`, `Figure 7`, or `Figure 8` strings were found in the final PDF text.
- Root paper source was synchronized to `attack_centric_can_ids_paper.tex`.

## Template note

The local TeX installation does not provide `llncs.cls`, so the source falls back to `article` for local compilation. The source still uses the provided LLNCS-compatible commands and will switch to `llncs` automatically when the class file is available.
