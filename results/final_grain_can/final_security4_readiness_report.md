# Final Security-Four Readiness Report Revision

**Assessment: not yet CCF A / Security Four-ready; suitable for CCF B or a measurement/protocol-gap submission with conservative claims.**

## Evidence Added
- Output integrity audit passed with zero detected empty/unreadable CSV/TEX/SVG problems.
- CT&T official sample-level Protocol B 2x negative-cap was completed for seeds 42/2024/2026 on key models.
- Protocol C 5x negative-cap was completed for seed 42 on key models.
- Full aggregate-window100 score dumps were generated for CT&T test01-test04.
- Low-FPR and approximate event-level metrics were recomputed from those score dumps.

## Key Results
- CT&T test02 GradientBoosting under 2x negative cap: F1=0.9655, AUROC=0.9981, AUPR=0.9362, FPR=0.0018.
- CT&T test02 GradientBoosting under 5x negative cap: F1=0.9655, AUROC=0.9981, AUPR=0.9362, FPR=0.0018.
- CT&T test04 GradientBoosting under 2x negative cap: F1=0.4803, AUROC=0.6100, AUPR=0.2619, FPR=0.0001.
- CT&T test04 GradientBoosting under 5x negative cap: F1=0.4796, AUROC=0.6094, AUPR=0.2649, FPR=0.0001.
- Aggregate-window test02 Recall@FPR upper bound remains strong; aggregate-window test04 has encouraging upper-bound score separability but lacks validation-threshold deployment proof.
- Event-level test04 recall from approximate aggregate-window boundaries: 0.3650 if the event table is accepted as approximate.

## Readiness Decision
- test02 is now well supported under larger negative caps.
- test04 is not solved: sample-level large-negative F1 is only moderate and event-level recall is weak.
- Leakage audit has no direct blocking leakage, but remaining statuses are {'pass': 9, 'partial': 3, 'not_used': 2, 'risk': 2, 'not_checked': 1}.
- Full-negative/chunked-full-negative is still missing.

**Recommendation:** do not claim Security-Four-level unknown-attack generalization. Submit as a protocol-gap / feature-preserving granularity study, or complete full-negative plus official event-boundary validation before aiming higher.
