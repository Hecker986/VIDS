# Normality Policy Analysis

This run only evaluates a post-hoc score-combination pilot because validation prediction dumps for all candidates are unavailable. Thresholds are marked as test upper bounds and must not be used as formal main results.

Findings:

1. This simple post-hoc normality score does not solve unknown attack. On CT&T test04, classifier-only F1=0.2907 while normality-only F1=0.0346.
2. Score combinations do not improve formal evidence because thresholds are test upper bounds and validation dumps are missing.
3. ROAD remains unchanged because the Transformer score is already the strongest available signal under this post-hoc policy.
4. A publishable normality policy would need train-normal-only calibration, validation-selected thresholds, and full prediction dumps; this pilot is not enough.
