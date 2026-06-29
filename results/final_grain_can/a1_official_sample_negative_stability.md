# A1 Official Sample-Level Negative Stability

Completed: Protocol A, 5 seeds, capped negative sampling. Protocols B/C/D were not rerun in this final packaging step and are listed in `missing_final_figures.md`.

Best completed rows:

```csv
dataset,model,f1_mean,f1_std,aupr_mean
ctt_test02,GradientBoosting,0.9649126180068643,0.0009301082661476407,0.9459407282859518
ctt_test01,GradientBoosting,0.8521190543418946,0.16898394192127877,0.8088365679437188
ctt_test03,GradientBoosting,0.657375165926924,0.04261757507113845,0.5443757866734771
ctt_test04,GradientBoosting,0.3883776726495897,0.16977148329329034,0.2422355206604559
```

test02 remains stable under 5 seeds. test04 has high variance and is not solved by sample-level ML alone.
