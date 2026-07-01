# Ranking Inversion

`predict_all_normal` rank by weighted-F1: 9; rank by attack-F1: 57.

This demonstrates that weighted-F1 can rank a non-detector near the top, while attack-centric metrics surface the actual IDS models.

Top by attack-F1:

```csv
method,kind,accuracy,weighted_f1,normal_f1,attack_f1,aupr,recall_at_fpr_1e_3,rank_by_accuracy,rank_by_weighted_f1,rank_by_normal_f1,rank_by_attack_f1,rank_by_aupr,rank_by_recall_at_fpr_1e_3,rank_gap_weighted_vs_attack,rank_inversion_magnitude
GradientBoosting / GRAIN_window_100,model,,,,0.6678067550235144,0.784533638387578,,53.0,53.0,53.0,1.0,2.0,1.0,-52.0,52.0
GradientBoosting / GRAIN_window_20,model,,,,0.641630351192002,0.7227739499342077,,53.0,53.0,53.0,2.0,3.0,1.0,-51.0,51.0
GradientBoosting / GRAIN_window_50,model,,,,0.6274193548387097,0.8235315589679248,,53.0,53.0,53.0,3.0,1.0,1.0,-50.0,50.0
GradientBoosting / GRAIN_window_10,model,,,,0.6001249888402821,0.6184722776882597,,53.0,53.0,53.0,4.0,4.0,1.0,-49.0,49.0
HistGradientBoosting / SAFE_CAN,model,0.9992834642721127,0.999195941327017,0.9996414217021972,0.5861692368179643,0.3671244738486416,,1.0,1.0,1.0,5.0,6.0,1.0,4.0,4.0
GradientBoosting / GRAIN_window_5,model,,,,0.5168307289874028,0.5140487763254977,,53.0,53.0,53.0,6.0,5.0,1.0,-47.0,47.0
GradientBoosting / SAFE_CAN,model,0.9991711391843988,0.9990267279617462,0.99958523826733,0.48120443139855523,0.2636050023882967,,2.0,2.0,2.0,7.0,11.0,1.0,5.0,5.0
GradientBoosting / SAFE_CAN,model,0.9991703071467122,0.9990246102478415,0.9995848225916162,0.47962427060109636,0.2646623611826009,,3.0,3.0,3.0,8.0,10.0,1.0,5.0,5.0
```
