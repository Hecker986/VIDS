# Paper Figure/Table Inventory

| Figure ID | Figure name | Input files | Output files | Recommended placement | Main message | Caveats |
|---|---|---|---|---|---|---|
| Figure 1 | CMF-CAN architecture | manual refined architecture | paper_fig1_architecture_refined.* | main paper | Core cross-modality pipeline | Post-processing branches intentionally omitted |
| Figure 2 | Main multi-dataset results | road_main_20ep.csv; ctt_generalization_15ep.csv; CrySyS optional | paper_fig2_main_multidataset_refined.* | main paper | Mixed main performance | Macro-F1 fallback to F1 where needed |
| Figure 3 | ROAD few-label | road_few_label_3seed_mean_std.csv | paper_fig3_road_few_label_refined.* | main paper | Label efficiency is mixed | Transformer wins several ROAD ratios |
| Figure 4 | CT&T few-label | ctt_few_label_3seed_mean_std.csv | paper_fig4_ctt_few_label_refined.* | main paper | Few-label is setting-dependent | Not stable dominance |
| Figure 5 | CT&T generalization | ctt_generalization_15ep.csv | paper_fig5_ctt_generalization_refined.* | main paper | Unknown settings are hard | test03/test04 low absolute F1 |
| Figure 6 | Ablation | road_ablation_20ep_merged.csv; ctt_ablation_15ep_merged.csv | paper_fig6_ablation_refined.* | main paper | Fusion components help in selected settings | Full not always best |
| Figure 7 | Recall@FPR | paper_table_low_fpr_summary.csv | paper_fig7*_recall_at_fpr*.{png,pdf,svg} | main/appendix | UV-KA low-FPR is strongest | Only measured budgets unless recomputed |
| Figure 8 | Efficiency trade-off | efficiency_road.csv; road_main_20ep.csv | paper_fig8_efficiency_tradeoff_refined.* | main/appendix | CMF-CAN has acceptable overhead | Slightly slower than Transformer |
| Figure 9 | Gate weights | results/cmf_predictions/*gate_weights.csv | paper_fig9_gate_weights.* | appendix | Gate interpretability from completed dumps | ROAD only until CT&T gates exist |
| Figure 10 | Per-attack results | results/cmf_predictions/*predictions.csv | paper_fig10_per_attack_results.* | appendix | Attack-level evidence from completed dumps | Attack labels follow processed dataset labels |
| Appendix | PR curves | results/cmf_predictions/*predictions.csv | paper_fig_pr_curves_road_ctt.* | appendix | Ranking behavior across shifted settings | Can be optimistic under threshold shift |
| Appendix | ROC curves | results/cmf_predictions/*predictions.csv | paper_fig_roc_curves_road_ctt.* | appendix | Ranking behavior across shifted settings | Use with low-FPR curves for deployment |
| Appendix | Failure cases | results/cmf_predictions/*predictions.csv | paper_fig_failure_cases.* | appendix | False positive/negative burden by setting | Counts depend on setting size |
| Appendix | Calibration reliability | results/cmf_predictions/*predictions.csv | paper_fig_calibration_reliability.* | appendix | Score calibration bins | Post-hoc calibration not applied here |
| Appendix | t-SNE embeddings | results/cmf_embeddings/*embedding_sample.* | paper_fig_tsne_embeddings.* | appendix | Representation separability sample | Sampled visualization only |

| Table | Output files | Recommended placement | Caveats |
|---|---|---|---|
| paper_table_dataset_summary_refined | `paper_table_dataset_summary_refined.csv`, `paper_table_dataset_summary_refined.tex` | main/appendix as appropriate | NA retained; no imputation |
| paper_table_overall_main_results_refined | `paper_table_overall_main_results_refined.csv`, `paper_table_overall_main_results_refined.tex` | main/appendix as appropriate | NA retained; no imputation |
| paper_table_few_label_refined | `paper_table_few_label_refined.csv`, `paper_table_few_label_refined.tex` | main/appendix as appropriate | NA retained; no imputation |
| paper_table_ctt_generalization_refined | `paper_table_ctt_generalization_refined.csv`, `paper_table_ctt_generalization_refined.tex` | main/appendix as appropriate | NA retained; no imputation |
| paper_table_low_fpr_refined | `paper_table_low_fpr_refined.csv`, `paper_table_low_fpr_refined.tex` | main/appendix as appropriate | NA retained; no imputation |
| paper_table_ablation_refined | `paper_table_ablation_refined.csv`, `paper_table_ablation_refined.tex` | main/appendix as appropriate | NA retained; no imputation |
| paper_table_efficiency_refined | `paper_table_efficiency_refined.csv`, `paper_table_efficiency_refined.tex` | main/appendix as appropriate | NA retained; no imputation |
