# A1 Official Sample Negative Stability Revision

Protocol A existing capped 5-seed rows are retained. This revision adds real Protocol B 2x negative-cap runs for seeds 42/2024/2026 and Protocol C 5x negative-cap seed 42 for the key sample-level models GradientBoosting, HistGradientBoosting, and LogisticRegression. Heavy ExtraTrees/RandomForest/MLP large-negative variants are marked resource-limited instead of fabricated. Protocol D full-negative remains not completed.

- CT&T test02 GradientBoosting, Protocol B: F1=0.9655, AUROC=0.9981, AUPR=0.9362, FPR=0.0018.
- CT&T test02 GradientBoosting, Protocol C: F1=0.9655, AUROC=0.9981, AUPR=0.9362, FPR=0.0018.
- CT&T test04 GradientBoosting, Protocol B: F1=0.4803, AUROC=0.6100, AUPR=0.2619, FPR=0.0001.
- CT&T test04 GradientBoosting, Protocol C: F1=0.4796, AUROC=0.6094, AUPR=0.2649, FPR=0.0001.

Interpretation: test02 remains stable under larger negative caps. test04 improves over the earliest official sample-level pilot but remains much weaker than test02 and cannot support an unknown-attack breakthrough claim.
