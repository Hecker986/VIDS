# Original Paper Gap Analysis

## What Was Compared

The public paper is **can-train-and-test: A Curated CAN Dataset for Automotive Intrusion Detection** by Lampe and Meng. The aligned arXiv version reports sub-dataset #1 / testing subset #4 in **Table 10**, while our task discussion has referred to this public high-score table as Table 13. We now treat the arXiv Table 10 values as the exact original-paper target for aligned `set_01` test04.

The original paper states that the benchmark tables show the best parameter combination for each model after sorting by F1-score. Therefore, our previous reproduction using small/default/local parameter settings was not a full parameter-grid reproduction of the paper.

## Why Our Attack-F1 Is Much Lower

The main gap is not explained by raw timestamp, public-default features, or risky protocol features. The stronger explanation is **metric definition and class imbalance**.

For test04, the positive attack rate in our aligned data is only:

```text
positive_rate = 0.001077
num_positive = 14,244
num_negative = 13,206,311
```

A trivial `predict_all_normal` baseline already gives:

```text
accuracy   = 0.998923
normal-F1  = 0.999461
weighted-F1= 0.998384
attack-F1  = 0.000000
```

So a score near 0.998 can be achieved while detecting zero attacks if the reported metric is accuracy, normal-positive F1, weighted-F1, or weighted recall.

## Evidence From Original Table Values

The original paper's test04 values have a revealing pattern:

- BIRCH: accuracy 0.9990, recall 0.9990, F1 0.9984
- GradientBoosting: accuracy 0.9984, recall 0.9984, F1 0.9982
- LogisticRegression: accuracy 0.9976, recall 0.9976, F1 0.9977
- MLP: accuracy 0.9973, recall 0.9973, F1 0.9976
- IsolationForest: accuracy 0.9881, recall 0.9881, F1 0.9930

This `accuracy == recall` pattern is not characteristic of attack-positive recall. It is exactly what happens with sklearn's weighted recall in a single-label classification problem: weighted recall equals accuracy.

## Metric Matching Results

The metric matching table is saved at:

```text
results/table13_metric_forensics/tables/table13_metric_matching.csv
```

Important rows:

- GradientBoosting original F1 0.9982 is closest to our reconstructed weighted-F1 / normal-F1 / accuracy-like metrics, not attack-F1.
- GaussianNB original F1 0.9411 is almost exactly matched by reconstructed weighted-F1 in one local public-default row.
- Original models with near-zero precision/recall/F1, such as RandomForest, ExtraTrees, LinearSVM and DecisionTree, are consistent with attack-positive-style failure rows.
- BIRCH remains unresolved: deployable validation-threshold BIRCH does not reproduce 0.998 attack-F1, and unsafe test-majority cluster mapping cannot be verified because cluster assignments were not saved.

## Why This Matters

The original benchmark likely mixes two interpretations:

1. Some high reported rows look like weighted/accuracy-like metrics dominated by the normal class.
2. Some low reported rows look like attack-positive failure rows.

Therefore, it is unsafe to compare our corrected attack-positive F1 directly against the public reported F1. The right comparison is:

- corrected attack-positive F1,
- attack precision,
- attack recall,
- AUPR,
- AUROC,
- low-FPR recall,
- event-level false alarm behavior.

## Corrected Benchmark Position

In the corrected benchmark:

- Best sample-level public-reproduction row: `SAFE_CAN + HistGradientBoosting + 5x negative`, attack-F1 = 0.5862.
- Best corrected test04 row currently available: `GradientBoosting / window_100`, attack-F1 = 0.6678.

This is substantially below 0.998, but it is measuring the harder and security-relevant quantity: attack detection.

## Next Concrete Reproduction Steps

To fully close the gap with the original paper, the next experiment should not be more model tweaking. It should reproduce the original evaluation code path:

1. Recreate or obtain the exact parameter grid used for each Table 10 model.
2. Run sklearn metrics with `average="weighted"`, `average="macro"`, and attack-positive binary metrics side by side.
3. Save confusion matrices for every model/parameter row.
4. For BIRCH/KMeans/MiniBatchKMeans, save cluster assignments and compare:
   - validation-only threshold/mapping,
   - train-majority mapping,
   - unsafe test-majority mapping.
5. Only after those checks should any public 0.998 number be interpreted as a true unknown-attack detection score.

## Paper Direction

The strongest current direction is a **CT&T metric/protocol correction paper**:

> Public high scores on CT&T test04 can be misleading under extreme class imbalance if weighted/accuracy-like metrics are interpreted as attack-detection performance. A corrected benchmark based on attack-positive F1, AUPR, low-FPR recall, and event-level false alarms gives a much harder and more security-relevant picture.

