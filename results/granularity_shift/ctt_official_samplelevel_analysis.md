# CT&T Official Sample-Level 5-Seed Analysis

Best per setting:

```csv
dataset,model,f1_mean,f1_std,aupr_mean,auroc_mean,fpr_mean
ctt_test02,GradientBoosting,0.9649126180068643,0.0009301082661476407,0.9459407282859518,0.998277110503329,0.0018452357270737601
ctt_test01,GradientBoosting,0.8521190543418946,0.16898394192127877,0.8088365679437188,0.8755417853308363,0.00136348080200158
ctt_test03,GradientBoosting,0.657375165926924,0.04261757507113845,0.5443757866734771,0.7500649073201824,0.00035491528768518
ctt_test04,GradientBoosting,0.3883776726495897,0.16977148329329034,0.2422355206604559,0.5925827110506519,0.00014974658706732
```

- test02 is considered solved at sample level if the 5-seed mean remains near the protocol_rescue pilot F1=0.9662.
- test04 remains unresolved unless its best mean clearly exceeds the previous GradientBoosting F1=0.3332.
- Prediction CSVs are deterministic 50k audit samples for seed 42 GradientBoosting/RandomForest; full per-frame dumps were omitted due size.
