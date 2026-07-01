# GRAIN Full Retraining Feature Ablation

Each row retrains a GradientBoosting GRAIN sample-level baseline with a different feature mask using all positives and capped negatives from the original CT&T train split, then evaluates on full CT&T test04. This is stricter than feature-importance proxy evidence, but it remains a sample-level ablation rather than the aggregate-window model.
