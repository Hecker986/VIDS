# Table 13 Metric Hypothesis Matrix

Overall hypotheses by local rows:

```text
overall_metric_hypothesis
weighted/accuracy-like    33
attack-positive-like      10
unknown_no_local_row       6
```

The repeated original-paper `accuracy == recall` pattern is explained by weighted recall, not by attack-positive recall. Rows with high reported F1 are generally closer to weighted/normal/accuracy-like metrics than attack-F1.
