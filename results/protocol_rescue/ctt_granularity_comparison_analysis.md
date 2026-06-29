# CT&T Granularity Comparison Analysis

Official sample-level ML is compared with existing window=100 deep-learning results. Short-window datasets were not materialized in this run and are explicitly marked NA.

Findings:

1. test01 is not a protocol problem: window=100 CMF-CAN reaches F1=0.9688 and Transformer reaches F1=0.9641, both above official sample-level ML in this pilot.
2. test02 is a protocol/granularity problem: official sample-level GradientBoosting reaches F1=0.9662, while current window=100 Transformer is F1=0.1600 and the best existing -Ctx ablation is F1=0.8528.
3. test03 improves strongly at sample level: RandomForest reaches F1=0.6666 versus window=100 CMF-CAN F1=0.0847.
4. test04 remains hard: sample-level GradientBoosting reaches F1=0.3332, better than most window models but still not publishable as a solved unknown vehicle + unknown attack result.
5. Short-window protocols remain missing. The next necessary experiment is to materialize window_size=10/20/50 and test whether they preserve the sample-level signal while reducing false alarms.
