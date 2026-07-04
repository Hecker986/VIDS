# Final Compile Report

Manuscript: `grain_can_revised.tex`

Compiled PDF: `grain_can_revised.pdf`

Template: LNCS `llncs.cls` copied from the provided conference-style template files.

Status:

- PDF compilation succeeded with `latexmk`.
- Page count: 8 pages, within the 16-page limit.
- Citation warnings: none after rerun.
- Overfull hbox warnings: none after final revision.
- Main-text scan found no ACE-CAN, CMF-CAN, Reliable-CMF-CAN, TFS-CAN, SOTA, or "unknown attack solved" claims.

Experiment evidence integrated:

- Conventional CT&T IDS metrics.
- Confusion matrices.
- Raw fixed-window versus GRAIN-CAN comparison.
- Rule/statistical baseline comparison recomputed on CT&T test04.
- Feature-group retraining ablation.
- Window-length sensitivity.
- Supplementary AUPR/AUROC/fixed-FPR metrics.

Interpretation boundary:

The revised paper frames GRAIN-CAN as a supervised feature-preserving CAN IDS baseline. It does not claim solved unknown-attack detection or universal cross-vehicle generalization.
