# CT&T All-Settings Corrected Benchmark

Corrected metrics show CT&T test04 remains the hardest shifted setting. GRAIN aggregate/window features dominate the corrected test04 attack-F1 leaderboard, while weighted-F1 can still make trivial baselines look strong.

Best row per setting:

```csv
dataset,setting,model,model_family,granularity,accuracy,weighted_f1,attack_precision,attack_recall,attack_f1,macro_f1,balanced_accuracy,mcc,auroc,aupr,recall_at_fpr_1e_4,recall_at_fpr_5e_4,recall_at_fpr_1e_3,recall_at_fpr_5e_3,recall_at_fpr_1e_2,source,num_pos,num_neg,positive_rate
ctt_test01,ctt_test01,CMF-CAN,window100 deep baseline,window100_deep,,,1.0,0.9394723228142784,0.9687916777807416,0.9838654431335369,,,0.9907451786831104,0.9781093030763408,,,,,,cmf_tables_overall_main_results_refined,1128,162419,0.0068988940321512
ctt_test02,ctt_test02,GradientBoosting / window_10,GRAIN-CAN,window_10,,,0.9959302210719718,0.99371180760891,0.9948197775957848,,,,0.9967256013347936,0.9901441834923794,,,,,,final_grain_can_b1_granularity_full_matrix,2137,168863,0.012497707013081102
ctt_test03,ctt_test03,GradientBoosting / window_100,GRAIN-CAN,window_100,,,0.9995979899497488,0.6622719403382608,0.7966997757129125,,,,0.9597628266647094,0.9249902867151728,,,,,,final_grain_can_b1_granularity_full_matrix,704,192168,0.0036489260522443
ctt_test04,ctt_test04,GradientBoosting / window_100,GRAIN-CAN,window_100,,,0.9364508393285372,0.518936877076412,0.6678067550235144,,,,0.9024154739312792,0.784533638387578,,,,,,final_grain_can_b1_granularity_full_matrix,638,238087,0.0026724392684081
```
