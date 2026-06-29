# Final CCF-A / Security-Four Direction Decision

Best observed sample-level row:

```csv
dataset,model,f1_mean,f1_std,aupr_mean,auroc_mean,fpr_mean
ctt_test02,GradientBoosting,0.9649126180068643,0.0009301082661476407,0.9459407282859518,0.998277110503329,0.0018452357270737601
```

1. The strongest current direction is official-feature sample/short-window GradientBoosting, not CMF/Reliable-CMF/TFS gate.
2. ROAD Transformer is still not beaten by CAN-Transformer+ under window=100; shifted CT&T test02 is robustly solved by official sample-level ML with 5-seed GradientBoosting F1=0.9649 ± 0.0009.
3. Feature-preserving short windows improve shifted CT&T further: test02 reaches F1=0.9948 at window_size=10, and test04 reaches F1=0.6678 at window_size=100 aggregate features.
4. This means the old failure was not "windowing" alone; it was the old deep window=100 representation. Feature-preserving aggregate windows retain the important timing/payload signals.
5. Normality policy is not the main solution in the current form. On test04 it only moves F1 from 0.33323 to 0.33363 and does not improve low-FPR recall.
6. The best CCF A/Security-Four direction is now: protocol-correct feature-preserving short/aggregate-window CAN IDS with event-level and low-false-alarm evaluation.
7. Remaining gap: prove results are stable under full-negative training or stronger sampling, add official event boundary/false-alarm-hour evaluation, and compare against stronger sequence/SSM baselines only after the protocol is fixed.
