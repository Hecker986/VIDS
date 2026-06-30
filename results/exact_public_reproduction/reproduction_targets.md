# Reproduction Targets

This task locks two separate targets. Exact reproduction is only valid if the dataset version, file manifest, feature protocol, model parameters and metric definition all align.

```csv
target_id,paper,dataset_version,repository,subset,test_setting,expected_model,expected_f1,expected_features,expected_metric
A,Lampe & Meng 2024 / can-train-and-test original paper,can-train-and-test original,brooke-lampe/can-train-and-test Bitbucket / original dataset release,Sub-dataset #1 / set_01,Testing subset #4 / unknown vehicle + unknown attack,BIRCH; GradientBoosting; LogisticRegression; MLP; IsolationForest,reported near 0.998 for selected Table-13-style benchmark rows per task statement,timestamp; arbitration ID; data field / payload variants,"F1, exact positive class definition must be verified"
B,can-sleuth 2025 / CT&T-v1.5 benchmark,can-train-and-test-v1.5,can-sleuth / can-train-and-test-v1.5 release,set_01,test04,MLP,0.9981 per task statement,timestamp; arbitration ID; data field subdivided/intact variants,"F1, exact averaging/positive-label definition must be verified"
```
