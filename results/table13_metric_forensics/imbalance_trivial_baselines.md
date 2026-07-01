# Imbalance Trivial Baselines

Test04 positive rate is 0.001077. Predict-all-normal has accuracy 0.998923, normal-positive F1 0.999461, weighted-F1 0.998384, and attack-positive F1 0.000000.

This confirms that accuracy/normal-class/weighted metrics can look excellent while detecting no attacks. Corrected benchmark reporting must center attack-positive F1, AUPR, AUROC and low-FPR recall.
