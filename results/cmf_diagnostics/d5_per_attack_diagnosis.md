# D5 Per-attack Diagnosis

## Hypothesis
Average F1 hides attack heterogeneity.

## Answers
- Best/worst attacks are listed in `d5_per_attack_diagnosis.csv`.
- Fuzzing/interval/systematic low recall often corresponds to very low mean scores, not just a threshold issue.
- Zero-recall attack groups usually have scores below the selected threshold; attack-specific calibration may help but cannot fix absent ranking.
- Gate summaries show which modality CMF-CAN uses per attack; use this for interpretation, not as causal proof.
