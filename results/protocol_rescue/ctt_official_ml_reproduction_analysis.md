# CT&T Official ML Reproduction Analysis

This run uses official CT&T frame-level CSVs and does not use the current window=100 deep-learning pipeline. Training keeps all sampled positives and a capped random subset of negatives; testing is full official test folders.

Best per setting:

```csv
dataset,model,f1,aupr,auroc,fpr,fnr
ctt_test02,GradientBoosting,0.966157964878348,0.9349964229025688,0.9980967425009822,0.0017750547045951,0.0019736000536039
ctt_test03,RandomForest,0.6665926809644036,0.5474522674684986,0.7890517019711785,0.0003720492892755,0.4231452581032413
ctt_test01,RandomForest,0.575446581930608,0.4240362815980054,0.7114855681985864,0.0005443851196671,0.5764538558786346
ctt_test04,GradientBoosting,0.3332298458035806,0.1424204094746703,0.4811710848656793,0.0001409931963589,0.7739399045212019
```

Findings:
1. CT&T test02 is recoverable under official sample-level features: GradientBoosting reaches F1=0.9662, AUROC=0.9981, AUPR=0.9350, FPR=0.0018. This is much stronger than the current window=100 Transformer/CMF-CAN behavior and shows the test02 failure is primarily a protocol/granularity issue.
2. CT&T test04 is not recovered: best F1 is only 0.3332 with GradientBoosting. This suggests unknown vehicle + unknown attack is not solved by simply returning to sample-level fields.
3. RandomForest is strongest for test01/test03, while GradientBoosting is strongest for test02/test04.
4. Current data version is sufficient to reproduce a high shifted result on test02, but not enough to reproduce a high test04 result under this pilot setup.
5. Because training negatives are sampled, this is a protocol rescue pilot, not a strict full-train public benchmark reproduction.
