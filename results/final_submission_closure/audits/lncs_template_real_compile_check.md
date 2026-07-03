# LNCS Template Real Compile Check

## Inputs checked

- Official class file supplied by the user: `llncs.cls`
- Official bibliography style supplied by the user: `splncs04.bst`
- Official sample file supplied by the user: `samplepaper.tex`
- Submission source checked against the template:
  `results/final_submission_closure/template/attack_centric_can_ids_paper_lncs_submission.tex`

The supplied files were copied into `results/final_submission_closure/template/` and used for a real LNCS compile in `results/lncs_template_check/`.

## Template alignment

- Document class matches the sample style: `\documentclass[runningheads]{llncs}`.
- Keywords use LNCS syntax: `\keywords{... \and ...}`.
- Acknowledgements and disclosure use the LNCS `credits` environment pattern, not numbered sections.
- The source keeps an inline `thebibliography`, which is allowed by the LNCS sample style.  `splncs04.bst` is included for template completeness if the bibliography is later moved to BibTeX.
- Font extraction support is enabled through T1 encoding and Latin Modern, avoiding ligature extraction problems in the PDF text.

## Compile results

- Official LNCS compile command:
  `latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=results/lncs_template_check results/lncs_template_check/attack_centric_can_ids_paper_lncs_submission.tex`
- Official LNCS PDF:
  `results/final_submission_closure/template/attack_centric_can_ids_paper_lncs_submission.pdf`
- Page count under official `llncs.cls` v2.24: 15 pages.
- File size: 477256 bytes.
- Undefined references: none.
- Undefined citations: none.
- Overfull boxes in official LNCS compile: none reported by the final grep check.
- Remaining log notes: two underfull bibliography lines for the Lampe and Meng reference.  These are harmless line-breaking warnings in the bibliography and do not affect layout correctness.

## Citation check

All three maintained TeX sources have 29 citation keys and 29 bibliography entries, with no missing or uncited bibliography keys:

- `attack_centric_can_ids_paper.tex`
- `results/paper_revision_strict_review/attack_centric_can_ids_paper_revised.tex`
- `results/final_submission_closure/template/attack_centric_can_ids_paper_lncs_submission.tex`

## Final status

The official-template submission source compiles successfully, remains within the 16-page limit, and is aligned with the provided LNCS template files.
