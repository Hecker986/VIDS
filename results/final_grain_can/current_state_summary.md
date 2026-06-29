# Current State Summary

Best per CT&T setting:

```csv
dataset,model,f1,f1_std,aupr,auroc,fpr,method_family,granularity,window_size,mean_attack_ratio,precision,recall,fnr,policy,alpha_supervised,threshold,recall_at_fpr_1em04,recall_at_fpr_5em04,recall_at_fpr_1em03
ctt_test02,GradientBoosting,0.9948197775957848,,0.9901441834923794,0.9967256013347936,0.0003467674873497,feature_preserving_granularity,window_10,10.0,0.3236093041592746,0.9959302210719718,0.99371180760891,0.00628819239109,,,,,,
ctt_test01,GradientBoosting,0.9685299295774648,,0.9413131660211788,0.972270347808666,0.0001885376368257,feature_preserving_granularity,window_20,20.0,0.3394121433168848,0.9941269482719676,0.944217978974469,0.055782021025531,,,,,,
ctt_test03,GradientBoosting,0.7966997757129125,,0.9249902867151728,0.9597628266647094,2.5368477130317865e-05,feature_preserving_granularity,window_100,100.0,0.0277320548674923,0.9995979899497488,0.6622719403382608,0.3377280596617392,,,,,,
ctt_test04,GradientBoosting,0.6678067550235144,,0.784533638387578,0.9024154739312792,0.0004055243123302,feature_preserving_granularity,window_100,100.0,0.0946445182724252,0.9364508393285372,0.518936877076412,0.481063122923588,,,,,,
```

- test02 is reliably solved by feature-preserving sample/short-window protocols, not by the old deep window=100 pipeline.
- test04 remains the core difficult setting, but feature-preserving aggregate windows improve it substantially over sample-level ML.
- Old CMF-CAN, Reliable-CMF-CAN, TFS gate, and old window=100 Transformer should be removed from the main method line and retained as baselines/negative evidence.
- The paper should be organized around Feature-Preserving Granularity-Aware CAN IDS.
