# Humanized revision report

## Draft rewrite pass

The manuscript was edited for a more natural technical-paper voice. The main changes were made in the abstract, introduction, related work, method description, results interpretation, discussion, and conclusion. The experimental claims and numerical results were not changed.

## What still sounded AI-generated before the final pass

- Several paragraphs used broad framing phrases such as "from first principles", "central claim", "broader lesson", and "strong evidence".
- The contribution list read like a generated outline rather than a paper-specific summary.
- Some results paragraphs stated significance before explaining the actual numbers.
- The method section used formulaic phrases such as "design principle" and "three useful properties".
- The discussion over-summarized the paper instead of stating limits plainly.

## Final rewrite choices

- Replaced broad claims with concrete statements tied to the CT&T test04 numbers.
- Removed formulaic negative parallelisms such as "not just".
- Reduced rule-of-three cadence in the evaluation and method sections.
- Rewrote the abstract and introduction in a direct reviewer-facing tone.
- Clarified that GRAIN-CAN is a feature-preserving baseline, not a new deep architecture.
- Kept the conservative boundaries: unknown attack is not solved, external datasets are sanity checks, and Table 13 remains a metric-ambiguity case rather than an accusation.
- Checked the revised TeX for em dashes, en dashes, common AI-signpost terms, and internal model residues.

## Verification

- Revised source: `attack_centric_can_ids_paper.tex`
- Revised copy: `results/paper_revision_strict_review/attack_centric_can_ids_paper_revised.tex`
- Compiled PDF: `results/paper_revision_strict_review/attack_centric_can_ids_paper_revised.pdf`
- Page count: 16
- Residual checks found no `CMF`, `Reliable`, `Concat`, `CAN-Tr`, `CAN-Transformer`, `TFS`, `N/A`, `Figure 7`, or `Figure 8` in the final PDF text.
