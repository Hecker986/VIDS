# D6 Window Label Dilution

## Hypothesis
Any-attack window labels dilute supervision when only a few frames are malicious.

## Answers
- Frame-level labels exist in processed parquet and were used to calculate attack-frame ratios.
- Check whether low-ratio buckets have lower recall in the CSV/figure.
- If CMF-CAN drops more sharply than Transformer, segment/top-k pooling or smaller windows are justified.
- If all models drop in low-ratio buckets, window construction is a root cause.
