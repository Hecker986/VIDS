# CMF-CAN execution status

Date: 2026-06-26

## Implemented

- Independent `cmf_can/` package for the CMF-CAN experiment.
- Three-modality feature builder:
  - frame-level sequence numeric features
  - window-level statistics
  - train-only ID context
- Models:
  - `cnn`, `lstm`, `gru`, `transformer`
  - `frame_only`, `stats_only`, `concat_fusion`
  - `cmf_can`
  - `cmf_can_robust` diagnostic variant with ID/modality dropout
- Metrics:
  - Accuracy, Precision, Recall, F1, Macro-F1
  - AUROC, AUPR
  - FPR, FNR
  - Recall/F1 at FPR <= 1e-4, 5e-4, 1e-3
- CT&T downloader and processor for:
  - `ctt_test01`: known vehicle + known attack
  - `ctt_test02`: unknown vehicle + known attack
  - `ctt_test03`: known vehicle + unknown attack
  - `ctt_test04`: unknown vehicle + unknown attack

## Data Completed

- ROAD full dataset:
  - `data/processed/road`
  - `data/processed/road/cmf_features`
- CT&T set_01 full download:
  - 52 CSV files
  - 1.6GB raw data
- CT&T processed:
  - `data/processed/ctt_test01`
  - `data/processed/ctt_test02`
  - `data/processed/ctt_test03`
  - `data/processed/ctt_test04`
- CT&T CMF features:
  - all four protocols completed

## Key Current Results

### ROAD 5 epoch key table

Source: `results/cmf_tables/road_main_5ep.csv`

- Transformer: F1 0.8159, AUROC 0.9377, FPR 1.30e-4.
- Stats-only: F1 0.2420, AUROC 0.6643.
- Concat-fusion: F1 0.6792, AUROC 0.8032.
- CMF-CAN standard: F1 0.7909, AUROC 0.8946, FPR 2.98e-4.

Interpretation: ROAD confirms frame sequence is essential. Standard CMF is viable but 5 epochs do not beat Transformer on default F1. It is close under low-FPR metrics.

### CT&T test01 5 epoch table

Source: `results/cmf_tables/ctt_generalization_5ep.csv`

- Transformer: F1 0.9011.
- Concat-fusion: F1 0.9370.
- CMF-CAN: F1 0.9417.

Interpretation: CMF-CAN improves known-vehicle/known-attack generalization.

### CT&T test02 diagnostics

Source: `results/cmf_tables/diagnostic_cmf_ctt.csv`

- Original CMF-CAN with unseen-ID mapping:
  - AUROC improved to 0.8140 but default F1 stayed 0.1600 due to threshold shift.
- Stats-only:
  - F1 0.8050, AUROC 0.9148.
  - Recall@FPR<=1e-4 0.8365.
- Robust CMF diagnostic:
  - AUROC 0.9750, AUPR 0.9437.
  - Default val-F1 threshold still fails.
  - Recall@FPR<=1e-4 0.8468, F1@FPR<=1e-3 0.9302.

Interpretation: Unknown-vehicle transfer is not solved by raw ID/sequence fusion. The ranking is good after robust training, but score calibration shifts across vehicles. CT&T unknown-vehicle results must emphasize AUROC/AUPR and constrained-FPR curves unless a valid calibration protocol is added.

### CT&T test02-test04 diagnostic batch

Source: `results/cmf_tables/ctt_generalization_diagnostic_5ep.csv`

- `ctt_test02` unknown vehicle + known attack:
  - `stats_only`: AUROC 0.9119, Recall@FPR<=1e-4 0.8377.
  - `cmf_can`: AUROC 0.8140, low-FPR recall very weak.
  - `cmf_can_robust`: AUROC 0.9750, Recall@FPR<=1e-4 0.8468, F1@FPR<=1e-3 0.9302.
- `ctt_test03` known vehicle + unknown attack:
  - `stats_only`: AUROC 0.8980, but low-FPR recall only 0.0473.
  - `cmf_can`: AUROC 0.6412.
  - `cmf_can_robust`: AUROC 0.7389.
- `ctt_test04` unknown vehicle + unknown attack:
  - `stats_only`: AUROC 0.6565.
  - `cmf_can`: AUROC 0.7154, F1@FPR<=1e-3 0.3004.
  - `cmf_can_robust`: AUROC 0.7572.

Interpretation: robust CMF fixes unknown-vehicle known-attack ranking, but unknown-attack generalization remains weak. This is a real limitation of the current method and should not be hidden.

### ROAD few-label seed=42 after model fixes

Source: `results/cmf_tables/road_few_label_seed42_20ep_merged.csv`

Configuration:

- Epochs: 20
- LR: 1e-4
- Seed: 42
- Models: Transformer, Concat-Fusion, CMF-CAN
- CMF fixes applied:
  - frame branch payload encoder aligned with Transformer baseline
  - auxiliary branch classification loss
  - frame-logit residual in CMF fusion

Key points:

- 1% labels:
  - Transformer F1 0.3077, AUROC 0.7653.
  - Concat-Fusion F1 0.4903, AUROC 0.8232.
  - CMF-CAN F1 0.5899, AUROC 0.8824.
  - This supports the label-efficient claim at the strictest label ratio.
- 5% labels:
  - Transformer F1 0.8107.
  - Concat-Fusion F1 0.7646.
  - CMF-CAN F1 0.7840, AUROC 0.9037.
  - CMF has better AUROC than Transformer but lower default F1.
- 10% labels:
  - Transformer F1 0.8035.
  - Concat-Fusion F1 0.8017.
  - CMF-CAN F1 0.7859, AUROC 0.9180.
- 20% labels:
  - Transformer F1 0.8147.
  - Concat-Fusion F1 0.7800.
  - CMF-CAN F1 0.7893, AUROC 0.9445.
  - CMF has the best AUROC and lowest FPR but not best default F1.
- 100% labels:
  - Transformer F1 0.8281.
  - Concat-Fusion F1 0.7832.
  - CMF-CAN F1 0.8039, AUROC 0.9386.

Interpretation: after fixes, CMF-CAN has a strong 1% label-efficiency result and strong AUROC/low-FPR behavior. It does not consistently beat Transformer on default F1 above 5%, so the paper claim should be framed around 1% labels, ranking quality, and low-FPR deployment metrics rather than universal F1 dominance.

### ROAD few-label 3-seed summary after model fixes

Source: `results/cmf_tables/road_few_label_3seed_mean_std.csv`

Completed seeds:

- 42
- 2024
- 2026

Configuration:

- Epochs: 20
- LR: 1e-4
- Models: Transformer, Concat-Fusion, CMF-CAN
- Label ratios: 1%, 5%, 10%, 20%, 100%

Mean results:

| Label ratio | Best default F1 | Best AUROC | Best AUPR | Best Recall@FPR<=1e-4 |
|---:|---|---|---|---|
| 1% | CMF-CAN 0.4655 | CMF-CAN 0.8108 | CMF-CAN 0.4597 | CMF-CAN 0.0686 |
| 5% | Transformer 0.8166 | CMF-CAN 0.9157 | CMF-CAN 0.7830 | CMF-CAN 0.5337 |
| 10% | Transformer 0.8150 | CMF-CAN 0.9158 | CMF-CAN 0.7751 | CMF-CAN 0.5712 |
| 20% | Transformer 0.8127 | CMF-CAN 0.9423 | CMF-CAN 0.8062 | CMF-CAN 0.6742 |
| 100% | Transformer 0.8257 | CMF-CAN 0.9460 | CMF-CAN 0.8135 | Concat-Fusion 0.7065 |

Interpretation:

- CMF-CAN is the best model for ranking quality across all ROAD label ratios by mean AUROC and AUPR.
- CMF-CAN is the best default-F1 model only at 1% labels. Above 5%, Transformer has the best mean default F1.
- CMF-CAN has the best mean Recall@FPR<=1e-4 from 1% through 20%, but Concat-Fusion slightly wins at 100% by a negligible margin.
- The paper should not claim universal default-F1 superiority on ROAD. The defensible claim is label efficiency at 1%, better ranking quality, and better low-FPR operation.

### ROAD full main table seed=42 after model fixes

Source: `results/cmf_tables/road_main_20ep.csv`

Configuration:

- Epochs: 20
- LR: 1e-4
- Seed: 42
- Label ratio: 100%
- Models: CNN, LSTM, GRU, Transformer, Stats-only, Concat-Fusion, CMF-CAN

Results sorted by default F1:

| Model | F1 | AUROC | AUPR | Recall@FPR<=1e-4 |
|---|---:|---:|---:|---:|
| Transformer | 0.8279 | 0.9335 | 0.7794 | 0.7069 |
| GRU | 0.8254 | 0.8746 | 0.7524 | 0.7069 |
| LSTM | 0.8240 | 0.8943 | 0.7604 | 0.0000 |
| CNN | 0.8206 | 0.9070 | 0.7544 | 0.0000 |
| CMF-CAN | 0.7894 | 0.9431 | 0.8060 | 0.6827 |
| Concat-Fusion | 0.7886 | 0.8849 | 0.7539 | 0.7061 |
| Stats-only | 0.2581 | 0.6721 | 0.2546 | 0.0841 |

Interpretation:

- Transformer is the best ROAD full-label default-F1 model.
- CMF-CAN is not competitive on default F1 in the full-label main table, but it has the best AUROC and AUPR.
- CMF-CAN's low-FPR recall is useful but not the best in this table.
- Stats-only is weak on ROAD, so the full model's gains do not come from statistics alone.

### ROAD ablation seed=42 after model fixes

Sources:

- `results/cmf_tables/road_ablation_20ep.csv`
- `results/cmf_tables/road_ablation_20ep_merged.csv`

Configuration:

- Epochs: 20
- LR: 1e-4
- Seed: 42
- Label ratio: 100%

Merged results:

| Model | F1 | AUROC | AUPR | Recall@FPR<=1e-4 |
|---|---:|---:|---:|---:|
| CMF-CAN | 0.7894 | 0.9431 | 0.8060 | 0.6827 |
| Frame-only | 0.8000 | 0.9347 | 0.8041 | 0.7058 |
| Stats-only | 0.2581 | 0.6721 | 0.2546 | 0.0841 |
| Concat-Fusion | 0.7886 | 0.8849 | 0.7539 | 0.7061 |
| w/o stats | 0.8089 | 0.9406 | 0.8004 | 0.7036 |
| w/o context | 0.8009 | 0.9078 | 0.7717 | 0.7018 |
| w/o cross-attention | 0.7859 | 0.9352 | 0.7960 | 0.7032 |
| w/o gate | 0.7987 | 0.9229 | 0.7852 | 0.7065 |

Interpretation:

- CMF-CAN has the best AUROC and AUPR among ablation variants.
- CMF-CAN does not have the best default F1 in the ablation table. `w/o stats`, `frame_only`, `w/o context`, and `w/o gate` have higher default F1.
- Removing cross-attention slightly lowers default F1 but improves Recall@FPR<=1e-4.
- The statistics branch is not a standalone ROAD detector and appears to trade default F1 for ranking quality when fused.
- The ablation claim should be framed as ranking-quality support, not as every module improving default F1.

### CT&T 15-epoch generalization

Source: `results/cmf_tables/ctt_generalization_15ep.csv`

Configuration:

- Epochs: 15
- LR: 1e-4
- Seed: 42
- Main models: Transformer, Concat-Fusion, CMF-CAN on `ctt_test01` to `ctt_test04`
- Diagnostic models: Stats-only and robust CMF on `ctt_test02` to `ctt_test04`

Best model by setting:

| Setting | Best default F1 | Best AUROC | Best AUPR | Best Recall@FPR<=1e-4 |
|---|---|---|---|---|
| `ctt_test01` known vehicle + known attack | CMF-CAN 0.9688 | Transformer 0.9969 | Transformer 0.9844 | Transformer 0.9715 |
| `ctt_test02` unknown vehicle + known attack | robust CMF 0.1619 | CMF-CAN 0.9684 | CMF-CAN 0.9325 | CMF-CAN 0.7680 |
| `ctt_test03` known vehicle + unknown attack | robust CMF 0.1242 | Stats-only 0.8584 | Stats-only 0.3882 | CMF-CAN 0.0832 |
| `ctt_test04` unknown vehicle + unknown attack | robust CMF 0.2109 | robust CMF 0.7387 | Stats-only 0.2940 | Stats-only 0.1581 |

Interpretation:

- CT&T `test01` supports the method: CMF-CAN has the best default F1.
- CT&T `test02` supports a low-FPR/ranking claim: CMF-CAN has the best AUROC, AUPR, and Recall@FPR<=1e-4, but the transferred default threshold fails.
- CT&T `test03` and `test04` remain weak. Unknown-attack generalization is not solved by the current method.
- Robust CMF at 15 epochs improves default F1 in `test03` and `test04`, but its `test02` low-FPR result is worse than the earlier 5-epoch diagnostic. This suggests over-training or calibration drift, so robust CMF should be reported as diagnostic rather than as the final main method without a validated early-stop/calibration rule.
- The defensible CT&T paper claim is: CMF-CAN works well on known setting and has strong unknown-vehicle ranking under constrained FPR; unknown-attack settings are a limitation.

### ROAD efficiency profile

Source: `results/cmf_tables/efficiency_road.csv`

Configuration:

- Dataset: ROAD test split
- Batch size: 512 windows
- Warmup: 5 batches
- Profile: 30 batches
- Device: CUDA

Results:

| Model | Params | Avg batch ms | Windows/s |
|---|---:|---:|---:|
| CNN | 492,802 | 2.27 | 225,666 |
| LSTM | 1,421,186 | 9.32 | 54,950 |
| GRU | 1,158,018 | 9.83 | 52,076 |
| Transformer | 1,949,826 | 8.25 | 62,060 |
| Frame-only | 2,703,179 | 9.91 | 51,687 |
| Stats-only | 2,703,179 | 1.15 | 443,688 |
| Concat-Fusion | 2,769,739 | 9.45 | 54,164 |
| CMF-CAN | 2,703,179 | 10.43 | 49,068 |

Interpretation:

- CMF-CAN is slower than Transformer but still processes about 49k windows/s at batch size 512 on the available GPU.
- CMF-CAN has about 2.7M instantiated parameters, larger than the Transformer baseline but still small enough for real-time batched inference.
- Stats-only appears to have the same instantiated parameter count because the current diagnostic implementation keeps unused modules in the object; its latency shows the active computation is much smaller.

### CT&T deployment low-FPR table

Source: `results/cmf_tables/ctt_deployment_low_fpr_15ep.csv`

Key results:

- `ctt_test01`: Transformer has the best Recall@FPR<=1e-4 at 0.9715; CMF-CAN is close at 0.9653 and has best default F1.
- `ctt_test02`: CMF-CAN has the best Recall@FPR<=1e-4 at 0.7680 and best AUROC/AUPR. This is the strongest CT&T generalization result for the method.
- `ctt_test03`: all models are weak under low FPR. CMF-CAN is highest at Recall@FPR<=1e-4, but only 0.0832.
- `ctt_test04`: Stats-only is best under low FPR, with Recall@FPR<=1e-4 of 0.1581. CMF-CAN is weak.

Interpretation: the deployment table should be separate from default-threshold F1. It supports a constrained-FPR claim for unknown-vehicle known-attack transfer, but not for unknown-attack transfer.

### CT&T few-label 3-seed summary

Sources:

- `results/cmf_tables/ctt_few_label_15ep.csv`
- `results/cmf_tables/ctt_few_label_3seed_mean_std.csv`

Configuration:

- Dataset: `ctt_test01`
- Epochs: 15
- Seeds: 42, 2024, 2026
- Label ratios: 1%, 5%, 10%, 20%, 100%
- Models: Transformer, Concat-Fusion, CMF-CAN

Mean results:

| Label ratio | Best default F1 | Best Recall@FPR<=1e-4 |
|---:|---|---|
| 1% | Transformer 0.7801 ± 0.1388 | Transformer 0.7479 ± 0.0347 |
| 5% | CMF-CAN 0.9045 ± 0.0252 | Transformer 0.7939 ± 0.0131 |
| 10% | Concat-Fusion 0.9231 ± 0.0193 | Transformer 0.8484 ± 0.0468 |
| 20% | Concat-Fusion 0.9422 ± 0.0235 | Transformer 0.8962 ± 0.0242 |
| 100% | Concat-Fusion 0.9367 ± 0.0218 | CMF-CAN 0.9690 ± 0.0014 |

Interpretation:

- CT&T few-label is not a strong universal CMF-CAN result.
- CMF-CAN is best by default F1 only at 5% labels, and best by strict low-FPR recall only at 100% labels.
- Several settings show high seed variance, especially 1% and CMF-CAN/Concat-Fusion. These results should be reported with mean ± std and discussed as unstable.
- The defensible few-label claim remains stronger on ROAD than on CT&T.

### CT&T test01 ablation

Sources:

- `results/cmf_tables/ctt_ablation_15ep.csv`
- `results/cmf_tables/ctt_ablation_15ep_merged.csv`

Configuration:

- Dataset: `ctt_test01`
- Epochs: 15
- Seed: 42
- Full CMF-CAN and Concat-Fusion rows reused from `ctt_generalization_15ep.csv`

Key results:

| Model | F1 | AUPR | AUROC | Recall@FPR<=1e-4 |
|---|---:|---:|---:|---:|
| CMF-CAN | 0.9688 | 0.9781 | 0.9907 | 0.9653 |
| Frame-only | 0.9776 | 0.9689 | 0.9755 | 0.9643 |
| Stats-only | 0.9420 | 0.9517 | 0.9819 | 0.8696 |
| Concat-Fusion | 0.8152 | 0.9755 | 0.9876 | 0.9555 |
| w/o stats | 0.9241 | 0.9743 | 0.9864 | 0.9503 |
| w/o context | 0.7485 | 0.9812 | 0.9944 | 0.9669 |
| w/o cross-attention | 0.9234 | 0.9789 | 0.9928 | 0.9659 |
| w/o gate | 0.7285 | 0.9835 | 0.9965 | 0.9690 |

Interpretation:

- CT&T test01 is easy enough that `frame_only` has the best default F1.
- CMF-CAN is competitive and has strong ranking/low-FPR behavior, but this ablation does not prove every module improves default F1.
- `w/o context` and `w/o gate` have poor default F1 but excellent AUROC/low-FPR recall, indicating threshold calibration problems rather than poor ranking.
- Strong ablation support should still lean on ROAD; CT&T ablation is useful mainly for showing calibration/ranking behavior.

## Issues Found And Fixed

- `window_stats` indexing bug:
  - val/test split used local split index instead of global window index.
  - This corrupted stats/concat/CMF validation and test.
  - Fixed in `cmf_can/data/dataset.py`.
- CT&T validation split bug:
  - naive last-20% time split produced no attack windows in val.
  - Fixed with paired-file split: `*-1.csv` train, `*-2.csv` val.
- Unknown CAN ID handling:
  - unseen test IDs previously used random untrained embeddings.
  - Fixed by mapping IDs with no train context to UNK index 4097.
- CT&T unknown-vehicle calibration issue:
  - not fully fixed.
  - robust CMF improves ranking/low-FPR metrics but not default val-F1 threshold transfer.

## Next Required Runs

1. ROAD few-label:
   - Completed for seeds 42, 2024, and 2026.
   - Use `results/cmf_tables/road_few_label_3seed_mean_std.csv` as the paper table source.
2. ROAD full main table:
   - Completed seed=42, 20 epochs.
   - Use `results/cmf_tables/road_main_20ep.csv` as the current main table source.
3. ROAD ablation:
   - Completed seed=42, 20 epochs.
   - Use `results/cmf_tables/road_ablation_20ep_merged.csv` as the current ablation source.
4. CT&T full generalization:
   - Completed 15-epoch formal table.
   - Use `results/cmf_tables/ctt_generalization_15ep.csv` as the current formal CT&T source.
   - Keep `results/cmf_tables/ctt_generalization_diagnostic_5ep.csv` for robust early-training diagnostics.
5. Decide calibration protocol:
   - Keep default val-F1 threshold for same-domain results.
   - For unknown-vehicle results, use the separate deployment table with low-FPR operating points.
   - Do not claim default F1 improvement on unknown vehicle until calibration is solved.
6. Efficiency:
   - Completed ROAD profiling.
   - Use `results/cmf_tables/efficiency_road.csv` as the efficiency source.
7. CT&T deployment low-FPR:
   - Completed from the 15-epoch CT&T table.
   - Use `results/cmf_tables/ctt_deployment_low_fpr_15ep.csv` as the deployment source.
8. CT&T few-label:
   - Completed for seeds 42, 2024, and 2026.
   - Use `results/cmf_tables/ctt_few_label_3seed_mean_std.csv` as the paper table source.
9. CT&T test01 ablation:
   - Completed seed=42.
   - Use `results/cmf_tables/ctt_ablation_15ep_merged.csv` as the paper table source.

## Exported Paper Artifacts

Generated by `python -m cmf_can.analysis.export_results --root /home/lqa/VIDS`.

Tables:

- `results/cmf_tables/table_road_main.tex`
- `results/cmf_tables/table_road_few_label.tex`
- `results/cmf_tables/table_road_ablation.tex`
- `results/cmf_tables/table_efficiency_road.tex`
- `results/cmf_tables/table_ctt_generalization.tex`
- `results/cmf_tables/table_ctt_deployment_low_fpr.tex`
- `results/cmf_tables/table_ctt_few_label.tex`
- `results/cmf_tables/table_ctt_ablation.tex`
- `results/cmf_tables/table_hcrl_main_15ep.tex`
- `results/cmf_tables/table_car_hacking_main_15ep.tex`
- `results/cmf_tables/table_crysys_subset_main_15ep.tex`
- `results/cmf_tables/table_crysys_family_mod_3seed.tex`

Figures:

- `results/cmf_figures/fig_road_main_aupr.png`
- `results/cmf_figures/fig_road_few_label_f1.png`
- `results/cmf_figures/fig_ctt_recall_at_fpr_1e4.png`
- `results/cmf_figures/fig_ctt_few_label_f1.png`
- `results/cmf_figures/fig_ctt_ablation_f1.png`
- `results/cmf_figures/fig_crysys_family_mod_recall_at_fpr_1e4.png`
