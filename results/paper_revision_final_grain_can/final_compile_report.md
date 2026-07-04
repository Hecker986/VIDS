# Final Compile Report

Manuscript: `grain_can_revised.tex`

Compiled PDF: `grain_can_revised.pdf`

Template: LNCS `llncs.cls` copied from the provided conference-style template files.

Status:

- PDF compilation succeeded with `latexmk`.
- Page count: 14 pages, within the 16-page limit.
- References: generated through BibTeX with `splncs04`; the final `.bbl` contains 20 cited entries, including recent CAN IDS / benchmark / evaluation work from 2022--2026.
- Citation warnings: none after rerun.
- Overfull hbox warnings: none after final revision.
- Main-text scan found no removed internal-model claims, no best-in-field wording, and no solved-unknown-attack overclaim.

Experiment evidence integrated:

- Conventional CT&T IDS metrics.
- Dataset/class-balance summary.
- Classifier, window-selection, and threshold-selection audits.
- Confusion matrices.
- Cross-shift decomposition.
- Raw fixed-window versus GRAIN-CAN comparison.
- Rule/statistical baseline comparison recomputed on CT&T test04.
- Feature-group retraining ablation.
- Window-length sensitivity.
- Supplementary AUPR/AUROC/fixed-FPR metrics.
- Metric-availability and prior-comparability audits.

Figure/table status:

- The revised manuscript includes the GRAIN-CAN method figure, local-evidence dilution motivation figure, conventional IDS results, raw-window comparison, rule/statistical baseline comparison, feature ablation, window sensitivity, and supplementary operating analysis.
- Figure 1 was redrawn as a three-stage feature-preserving IDS pipeline with non-overlapping text and fixed-protocol constraints.
- Float placement uses section barriers and `[!htbp]` so figures/tables are placed near their discussion instead of being repeated or dumped at the end.
- Tables are generated from CSV evidence and scaled to LNCS text width to avoid overflow.

Interpretation boundary:

The revised paper frames GRAIN-CAN as a supervised feature-preserving CAN IDS baseline. It does not claim solved unknown-attack detection or universal cross-vehicle generalization.
