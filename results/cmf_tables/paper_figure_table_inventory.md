# Paper Figure/Table Inventory

## Generated Figures
- `paper_fig1_architecture.png/.pdf/.svg`: input `results/cmf_figures/fig_model_architecture_cmf_can.*`; placement: 正文; note: Model architecture.
- `paper_fig2_main_multidataset.png/.pdf/.svg`: input `road_main_20ep.csv, ctt_generalization_15ep.csv, hcrl_main_15ep.csv, car_hacking_main_15ep.csv, crysys_family_mod_3model_3seed_mean_std.csv`; placement: 正文; note: Main multi-dataset performance.
- `paper_fig3_road_few_label.png/.pdf/.svg`: input `road_few_label_3seed_mean_std.csv`; placement: 正文; note: ROAD few-label curve.
- `paper_fig4_ctt_few_label.png/.pdf/.svg`: input `ctt_few_label_3seed_mean_std.csv`; placement: 正文; note: CT&T few-label curve.
- `paper_fig5_ctt_generalization.png/.pdf/.svg`: input `ctt_generalization_15ep.csv`; placement: 正文; note: Known/unknown generalization.
- `paper_fig6_ablation.png/.pdf/.svg`: input `road_ablation_20ep_merged.csv, ctt_ablation_15ep_merged.csv`; placement: 正文; note: Ablation study.
- `paper_fig7_recall_at_fpr.png/.pdf/.svg`: input `ctt_deployment_low_fpr_15ep.csv`; placement: 正文/附录; note: Low-FPR deployment recall.
- `paper_fig8_efficiency_tradeoff.png/.pdf/.svg`: input `efficiency_road.csv, road_main_20ep.csv`; placement: 正文/附录; note: Efficiency-performance trade-off.
- `paper_fig_appendix_ood_score_summary.png/.pdf/.svg`: input `ood_score_summary.csv`; placement: 附录; note: Summary bar only; no distribution without per-sample scores.

## Generated Tables
- `paper_table_dataset_summary.csv/.tex`
- `paper_table_overall_main_results.csv/.tex`
- `paper_table_few_label_summary.csv/.tex`
- `paper_table_ctt_generalization_summary.csv/.tex`
- `paper_table_low_fpr_summary.csv/.tex`
- `paper_table_efficiency_summary.csv/.tex`
- `paper_table_calibration_summary.csv/.tex`
- `paper_table_ood_score_summary.csv`

## Not Generated Because Inputs Are Missing
- PR/ROC curves: no per-sample score/probability and label dump.
- Per-attack recall/F1: no per-sample prediction with attack_type.
- Gate weights by attack/setting: no saved gate weights.
- UMAP/t-SNE: no embedding dump.
- Failure case analysis: no per-sample prediction dump.
- Reliability curve: no bin-level calibration data.
- OOD score distribution: no per-sample OOD scores.
