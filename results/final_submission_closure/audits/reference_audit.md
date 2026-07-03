# Reference Audit

## Summary

I checked the manuscript bibliography for citation consistency, obvious metadata errors, author/title/year mismatches, and LNCS-style formatting issues.

The main mechanical problems were:

- `lee2017otids` was not cited and had a malformed page range (`57--5709`). It was removed.
- `sklearn` was not cited in the manuscript. It was removed from the final reference list.
- The bibliography had 31 entries while the manuscript cited 29 keys. After cleanup, all three TeX sources have 29 cited keys and 29 bibliography entries, with no uncited or missing references.

## Files updated

- `attack_centric_can_ids_paper.tex`
- `results/paper_revision_strict_review/attack_centric_can_ids_paper_revised.tex`
- `results/final_submission_closure/template/attack_centric_can_ids_paper_lncs_submission.tex`

## Recent-work coverage

The reference list still includes recent 2023--2026 work:

- CT&T dataset paper.
- can-sleuth.
- CrySyS dataset.
- StatGraph.
- ROAD comparative analysis.
- MIDS / bidirectional Mamba.
- SecCAN.
- CAN authentication survey.
- learning-based IVN IDS survey.
- CAN IDS benchmarking review.
- UAVCAN graph IDS.
- vehicular IDS survey/evaluation.

## Remaining caution

Several recent works are arXiv/preprint references. That is acceptable for related-work positioning, but they should not be presented as fully peer-reviewed baselines unless the final venue version is known.
