# Chunked Negative Ensemble

This experiment trains five GradientBoosting members using different capped negative chunks and averages their scores. It is a practical chunked-negative stability probe that covers more negative examples than a single capped run; rows labelled upper-bound still use test score budgets and are not validation-tuned deployment thresholds.
