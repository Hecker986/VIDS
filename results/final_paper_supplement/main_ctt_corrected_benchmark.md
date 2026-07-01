# Main CT&T Corrected Benchmark

Weighted-F1 and attack-F1 give different conclusions whenever all-normal or normal-dominated rows are present. CT&T test04 remains the hardest setting under attack-centric metrics. GRAIN_window_100 is the strongest corrected test04 baseline in this table, but this does not support an unknown-attack-solved claim.

Best measured rows:

```csv
setting,model,accuracy,weighted_f1,attack_f1,attack_precision,attack_recall,aupr,auroc,recall_at_fpr_1e_3,positive_rate,source,support_status
ctt_test01,CMF-CAN,,,0.9687916777807416,1.0,0.9394723228142784,0.9781093030763408,0.9907451786831104,,0.0068988940321512,attack_centric_final_or_ctt_generalization,measured
ctt_test02,GRAIN_window_10,,,0.9948197775957848,0.9959302210719718,0.99371180760891,0.9901441834923794,0.9967256013347936,,0.0124977070130811,attack_centric_final,measured
ctt_test03,GRAIN_window_100,,,0.7966997757129125,0.9995979899497488,0.6622719403382608,0.9249902867151728,0.9597628266647094,,0.0036489260522443,attack_centric_final,measured
ctt_test04,GRAIN_window_100,,,0.6678067550235144,0.9364508393285372,0.518936877076412,0.784533638387578,0.9024154739312792,,0.0026724392684081,attack_centric_final,measured
```
