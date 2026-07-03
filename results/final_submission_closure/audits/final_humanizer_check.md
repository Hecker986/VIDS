# Final humanizer check

## Scope

Checked and edited:

- `attack_centric_can_ids_paper.tex`
- `results/paper_revision_strict_review/attack_centric_can_ids_paper_revised.tex`
- `results/final_submission_closure/template/attack_centric_can_ids_paper_lncs_submission.tex`

The edit focused on prose only. No experiment numbers, table values, figure references, or conclusions were changed.

## Changes made

- Replaced a few broad or promotional phrases with plainer technical wording.
- Removed some defensive phrasing from the discussion.
- Reworded the detector/benchmark relationship so it reads like an authorial argument rather than a generated summary.
- Kept the claim boundary intact: the paper still says CT&T test04 is not solved.

## Final checks

- Revised PDF pages: 16.
- Figures: 5.
- Tables: 5.
- Bibliography entries: 29.
- Cited keys: 29.
- Uncited bibliography entries: 0.
- Missing bibliography entries: 0.
- No unsafe phrases found in PDF text:
  - `CMF`
  - `Reliable`
  - `Concat`
  - `CAN-Tr`
  - `N/A`
  - `unknown attack solved`
  - `public 0.998 is attack-F1`
  - `CT&T dataset is wrong`
  - `original paper is wrong`
- No Unicode em dash, en dash, or curly quotes found in the checked TeX files.

## Remaining warnings

Two small overfull boxes remain in the local fallback build. They are minor and do not affect compilation. The official LLNCS build should be checked again once `llncs.cls` is available.
