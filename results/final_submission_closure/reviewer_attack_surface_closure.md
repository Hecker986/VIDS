# Reviewer Attack-Surface Closure

This file summarizes the likely reviewer attacks that would still matter before submission and how the current evidence package handles them.

## Closed by evidence

- Metric trap: closed by the attack-centric metric audit, Table 13 metric forensics, ranking inversion, and all-normal baseline evidence.
- Internal prototype contamination: closed in the manuscript. CMF-CAN, Reliable-CMF-CAN, Concat, and CAN-Transformer shorthand rows are not present in the final PDF text.
- GRAIN-CAN definition: closed by the method section and Algorithm 1. The paper now defines GRAIN-CAN as a feature-preserving representation plus lightweight score-producing classifier, not as a new deep architecture.
- Page limit and figure/table cleanup: the locally compiled revised PDF is 16 pages. The strict LNCS source is generated separately for official-template compilation.

## Contained by conservative claims

- Unknown attacks are not claimed solved.
- CT&T and prior authors are not accused of being wrong.
- Best-test rows are labelled diagnostic score-separability evidence.
- Event-level rows are labelled approximate where official boundaries are unavailable.
- External datasets are sanity checks, not evidence of universal dominance.

## External blockers that cannot be fabricated

- Original author confusion matrices and exact code.
- Completely aligned can-train-and-test-v1.5 data/manifest.
- Official event boundaries.

These are documented as auditable blockers rather than hidden omissions. The manuscript should not claim exact public-result alignment or official event-level deployment evidence until these artifacts are obtained.
