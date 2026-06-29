# CMF-CAN Core Goal Completion Report

Date: 2026-06-27

## Final Position

The core goal is considered complete for a publishable prototype when the paper claims are scoped as low-label, low-FPR CAN intrusion detection with cross-modality fusion and an anomaly-aware branch for shifted or unknown attacks.
It is not complete for an over-claim that every dataset, every shift, and a 0.1% supervised-label setting are solved by one classifier.

## Academic and Industrial Method Mapping

| Method | Role | Used for | Status | Reference |
|---|---|---|---|---|
| Temperature/Platt calibration | same-domain probability calibration and ECE reduction | CT&T test01, ROAD 5% | implemented | Guo et al., On Calibration of Modern Neural Networks, 2017, https://arxiv.org/abs/1706.04599 |
| Energy/OOD scoring | post-hoc open-set score for distribution shift | CT&T test02-test04 diagnostics | implemented; useful but insufficient alone | Liu et al., Energy-based Out-of-distribution Detection, 2020, https://arxiv.org/abs/2010.03759 |
| One-class normality modeling | benign-only anomaly detection for unknown attacks | CMF-CAN+Anomaly | implemented | one-class neural anomaly detection family, e.g. https://arxiv.org/abs/1802.06360 |
| Risk-controlled thresholding | validation-calibrated low-FPR operating points | deployment policy table | implemented with validation constrained-FPR thresholds | split conformal / risk-control principle; automotive risk context: UNECE R155 |

| Industry requirement | Evidence in this project | Status |
|---|---|---|
| lifecycle cybersecurity and monitoring | final policy reports operating points and failure boundaries | covered for paper prototype |
| low false-positive operation | Recall/F1 at FPR <= 1e-4, 5e-4, 1e-3 are reported | covered |
| risk-aware deployment rather than universal threshold | policy selects different score families for known-domain, shifted-domain, and unknown-attack settings | covered |
| known limitations and fallback | CT&T test04 and CrySyS subset limitations are explicitly stated | covered with caveats |

## Final Deployment Policy

| Dataset | Label ratio | Score | Threshold policy | F1 | AUPR | Recall@FPR<=1e-4 | Claim |
|---|---:|---|---|---:|---:|---:|---|
| ctt_test01 | 1.0000 | model_attack_prob | val_fpr_1em04 | 0.9819 | 0.9826 | 0.9679 | known vehicle + known attack; deploy calibrated neural score |
| ctt_test02 | 1.0000 | per-seed extended anomaly policy | validation-selected low-FPR/F1 policy | 0.3329 | 0.9275 | 0.8861 | unknown vehicle + known attack; extended anomaly fusion fixes threshold transfer while preserving ranking |
| ctt_test03 | 1.0000 | per-seed extended anomaly policy | validation-selected low-FPR/F1 policy | 0.8567 | 0.8457 | 0.2518 | known vehicle + unknown attack; Ledoit-Wolf normality score is the primary repair |
| ctt_test04 | 1.0000 | per-seed extended anomaly policy | validation-selected low-FPR/F1 policy | 0.4457 | 0.2992 | 0.1329 | unknown vehicle + unknown attack; extended anomaly branch significantly improves but does not fully solve the hardest setting |
| road | 0.0100 | model_attack_prob | val_f1 | 0.5899 | 0.5974 | 0.0932 | few-label ROAD; keep CMF-CAN discriminative score, do not use anomaly branch |
| road | 0.0500 | model_attack_prob | val_fpr_1em03 | 0.7971 | 0.7586 | 0.4653 | ROAD label efficiency / ranking; use CMF-CAN score with validation low-FPR threshold |

## Calibration Evidence

| Dataset | Label ratio | Raw F1 | Best policy F1 | Raw ECE | Best ECE |
|---|---:|---:|---:|---:|---:|
| ctt_test01 | 1.0000 | 0.8012 | 0.9824 | 0.0249 | 0.0032 |
| ctt_test02 | 1.0000 | 0.1600 | 0.1600 | 0.0868 | 0.0868 |
| ctt_test03 | 1.0000 | 0.0847 | 0.1563 | 0.0870 | 0.0746 |
| ctt_test04 | 1.0000 | 0.0282 | 0.0282 | 0.0114 | 0.0073 |
| road | 0.0100 | 0.4655 | 0.4655 | 0.0773 | 0.0696 |
| road | 0.0500 | 0.7891 | 0.8032 | 0.0164 | 0.0106 |

## Completion Assessment

| Core target | Completion | Evidence | Remaining caveat |
|---|---:|---|---|
| Cross-modality CAN representation | 100% | frame/window/context encoders and fusion ablations exist | none for current task scope |
| Low-label detection claim | 100% if scoped to 1%/5%; not 100% for 0.1% supervised labels | ROAD 1% and 5% 3-seed tables | do not claim 1.0 per mille supervised success without a new setting |
| Low-FPR deployment reporting | 100% | constrained-FPR metrics, calibration, and final policy table | deployment still requires vehicle-specific validation |
| Unknown-attack improvement | 100% for demonstrating improvement; not 100% for fully solving all unknown attacks | CT&T test03/test04 anomaly gains | test04 remains a hard/open setting |
| Industrial-style risk framing | 100% for paper prototype | explicit low-FPR policy and limitations | not a certified ISO/SAE 21434 implementation |

## Non-Negotiable Paper Wording

- Do not claim universal F1 superiority.
- Do not claim full CrySyS if only subset/family subset results are used.
- Do not claim 1.0 per mille supervised learning unless 0.001 label-ratio experiments are redesigned around benign-only/self-supervised learning and completed.
- Claim CMF-CAN as a cross-modality low-label/ranking detector, and CMF-CAN+Anomaly as the unknown-attack/shifted-domain extension.
