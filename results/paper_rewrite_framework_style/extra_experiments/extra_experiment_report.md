# Extra Experiment Report

All experiments use CT&T set_01, attack as the positive class, fixed W100 unless otherwise stated, and validation-F1 threshold selection. No test labels are used to select features, windows, classifiers, or thresholds.

## Completed experiments
- raw_same_classifier_control: 8 rows
- efficiency_cost: 2 rows
- cross_shift_matrix: 12 rows
- low_fpr_operating_curve: 45 rows
- feature_ablation_extended: 24 rows
- grain_classifier_sensitivity: 10 rows

## Key numeric checks
- Test01 same-classifier F1: Raw-window+GB=0.9677, GRAIN+GB=0.9423, delta=-0.0254.
- Test02 same-classifier F1: Raw-window+GB=0.9444, GRAIN+GB=0.9568, delta=+0.0124.
- Test03 same-classifier F1: Raw-window+GB=0.1344, GRAIN+GB=0.7958, delta=+0.6614.
- Test04 same-classifier F1: Raw-window+GB=0.2333, GRAIN+GB=0.7619, delta=+0.5286.
- Test04 classifier sensitivity best row: LogisticRegression on fixed GRAIN-W100, F1=0.8469, AUPR=0.9148.

## Evidence impact
- Raw-window+same-classifier control isolates representation from classifier choice.
- Efficiency measurements support or qualify the lightweight deployment claim using CPU-only timing.
- Cross-shift matrix treats Test01-Test04 as vehicle-shift by attack-shift categories, not a sequence.
- Low-FPR curves are diagnostic score operating curves for score-producing models only.
- Feature ablation extension retrains each mask; it should be read as setting-specific evidence.
- Classifier sensitivity fixes GRAIN-W100 features and varies classifier family.
- The classifier sensitivity result should not silently replace the main protocol; it is a new candidate backed by this supplement and should be introduced explicitly if used.

## Main-text suitability
- P0 raw/same-classifier and cross-shift matrix can go in the main paper or appendix depending on page budget.
- Efficiency belongs in an appendix or short deployment-cost paragraph.
- P1/P2 rows are best suited for appendix unless a reviewer asks specifically for them.
