# Metric Trap Audit Across Datasets

The trap is not unique to CT&T test04, but CT&T test04 is among the most extreme rare-attack settings. Rows with very low positive rate allow all-normal predictions to obtain high accuracy/weighted-F1 while attack-F1 remains zero.

Most severe rows:

```csv
dataset,num_pos,num_neg,positive_rate,predict_all_normal_accuracy,predict_all_normal_weighted_f1,predict_all_normal_normal_f1,predict_all_normal_attack_f1,predict_all_attack_attack_f1,imbalance_severity_score,metric_trap_risk_level,source,notes
ctt_test04_sample_level,14244,13206311,0.0010774131645759199,0.9989225868354241,0.9983841706143375,0.9994610030565108,0.0,0.002152507189568954,2.967617722248622,critical,metric_correction_full_metrics,sample-level class totals
ctt_test04,638,238087,0.0026724392684081,0.9973274688449052,0.9959929912622877,0.9986619464275228,0.0,0.005330815539577963,2.5730921554156967,critical,paper_table_dataset_summary_refined,nan
ctt_test03,704,192168,0.0036489260522443,0.9963499108216849,0.9945282031102303,0.9981716185331394,0.0,0.00727362896226805,2.4378349377939332,critical,paper_table_dataset_summary_refined,nan
ctt_test01,1128,162419,0.0068988940321512,0.993102900083768,0.9896662837762559,0.99653951639128,0.0,0.013699711553059055,2.1612205258180723,high,paper_table_dataset_summary_refined,nan
ctt_test02,2137,168863,0.012497707013081102,0.9875029239766082,0.9812936756955714,0.993712172257645,0.0,0.024685653557587343,1.9031696608250857,high,paper_table_dataset_summary_refined,nan
```
