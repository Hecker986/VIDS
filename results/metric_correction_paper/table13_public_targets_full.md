# Table 13 / Original Table 10 Public Targets

Rows with `accuracy == recall` within 1e-4: 9 / 14.

This repeated equality is consistent with sklearn weighted recall in single-label classification, because weighted recall equals accuracy.

```csv
model,reported_accuracy,reported_recall,reported_f1
GaussianNB,0.8906,0.8906,0.9411
LogisticRegression,0.9976,0.9976,0.9977
GradientBoosting,0.9984,0.9984,0.9982
IsolationForest,0.9881,0.9881,0.993
KMeans,0.8872,0.8872,0.9392
MiniBatchKMeans,0.4748,0.4748,0.6429
BIRCH,0.999,0.999,0.9984
MLP,0.9973,0.9973,0.9976
RestrictedBoltzmannMachine,0.001,0.001,0.0
```
