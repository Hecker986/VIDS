# Paper Revision After Final Evidence Completion

## Scope

This revision updates the paper after the final evidence completion experiments. The main source files are:

- `attack_centric_can_ids_paper.tex`
- `results/paper_revision_strict_review/attack_centric_can_ids_paper_revised.tex`
- `results/paper_revision_strict_review/attack_centric_can_ids_paper_revised.pdf`

## Evidence Added to the Paper

1. Validation-threshold low-FPR evidence is now reflected in the abstract, contributions, RQ5, Table 6, and limitations.
   - CT&T test04 GRAIN window100 validation FPR budget 5e-4: recall 0.8053, actual FPR 0.000681, AUPR 0.7868.

2. Large-negative heavy-model completion is now reflected in the contributions and evidence-closure section.
   - 48 B/C rows completed for ExtraTrees, RandomForest, and MLP.
   - Exact Protocol D full-negative remains a distinct exhaustive protocol marker, not a claimed completed result.

3. Multi-seed feature-removal evidence is now reflected in RQ4.
   - Timing-only mean attack-F1 0.6013.
   - Removing delta_t_same_id drops mean attack-F1 to about 0.0085.

4. External dependency audits are now framed as protocol boundaries, not as missing unhandled work.
   - Original confusion matrices/exact code, v1.5 exact alignment, and official event boundaries are documented as audited external dependencies.

## New Figure

Added:

- `results/paper_revision_strict_review/figures/figure8_evidence_closure.svg`
- `results/paper_revision_strict_review/figures/figure8_evidence_closure.pdf`

The figure has three panels:

1. Validation-threshold low-FPR recall across CT&T test01-test04.
2. Feature-removal retraining evidence.
3. Evidence-closure status summary.

## Consistency Checks

- All LaTeX figure paths resolve to existing PDF files.
- Recompiled revised PDF successfully.
- `pdftotext` check found no `N/A`, `CMF-CAN`, or `Reliable-CMF` residue in the revised PDF.
- Remaining `not found` language is limited to audited external dependency boundaries.
- The current fallback compile uses `article` because `llncs.cls` is not present in the repository. The source keeps the LLNCS path and will use `llncs.cls` automatically if placed in the project root.

## Current PDF

- Output: `results/paper_revision_strict_review/attack_centric_can_ids_paper_revised.pdf`
- Fallback compile page count: 21 pages.
- This page count is from the local fallback article class, not the official LNCS class.
