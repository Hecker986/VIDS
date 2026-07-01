# Recommended Paper Outline

Title: Metric Forensics for Vehicle-Shifted CAN Intrusion Detection: Correcting CT&T Test04 Evaluation

Abstract claim: Public CT&T test04 high scores can be dominated by normal-class weighted metrics; attack-centric correction reveals a harder benchmark.

Intro problem: unknown-vehicle/unknown-attack CAN IDS evaluation is high-stakes but metric ambiguity can invert conclusions.

Threat to validity: do not accuse public authors; state metric ambiguity, reproduction limits, and need for confusion matrices.

Method / metric forensics: compare original table values, weighted/normal/attack metrics, trivial baselines, and ranking inversion.

Corrected benchmark: report attack-F1, AUPR, AUROC, low-FPR recall and event evidence.

Main results: weighted/accuracy-like metrics can reach 0.998 with no attacks detected; corrected best attack-F1 remains far lower.

Discussion: propose corrected reporting standard for shifted CAN IDS.

Limitations: exact v1.5/code/parameter grids and official event boundaries are still needed.
