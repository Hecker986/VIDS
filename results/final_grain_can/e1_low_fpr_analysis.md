# E1 Low-FPR Revision

Low-FPR metrics were recomputed from newly generated full aggregate-window100 score dumps for CT&T test01-test04. The rows are threshold-budget upper bounds computed on the score dumps; formal validation-budget thresholds still require validation score dumps and must not be presented as official deployment thresholds.

Key aggregate-window recall values:

```text
fpr_budget                                    0.0001  0.0005  0.0010  0.0050  0.0100
dataset    model            granularity                                             
ctt_test02 GradientBoosting aggregate_window  0.6469  0.9301  0.9306  0.9306  0.9306
ctt_test04 GradientBoosting aggregate_window  0.4478  0.6246  0.8053  0.8053  0.8053
```

Interpretation: CT&T test02 retains strong low-FPR recall. CT&T test04 has encouraging score separability in the upper-bound budget analysis, but because validation-threshold rows are absent, it is not yet deployable evidence.
