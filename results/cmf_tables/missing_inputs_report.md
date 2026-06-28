# Missing Inputs Report

No fake figures were generated. The following missing or partial inputs remain:

- Missing prediction dump: ctt_test02_transformer_predictions.csv
- Missing prediction dump: ctt_test02_concat_fusion_predictions.csv
- Missing prediction dump: ctt_test02_cmf_can_predictions.csv
- Missing prediction dump: ctt_test03_transformer_predictions.csv
- Missing prediction dump: ctt_test03_concat_fusion_predictions.csv
- Missing prediction dump: ctt_test03_cmf_can_predictions.csv
- Missing prediction dump: ctt_test04_transformer_predictions.csv
- Missing prediction dump: ctt_test04_concat_fusion_predictions.csv
- Missing prediction dump: ctt_test04_cmf_can_predictions.csv
- Missing gate weight dump: ctt_test02_cmf_can_gate_weights.csv
- Missing gate weight dump: ctt_test03_cmf_can_gate_weights.csv
- Missing gate weight dump: ctt_test04_cmf_can_gate_weights.csv
- Missing embedding dumps for UMAP/t-SNE.
- Missing CT&T unknown-setting ablation CSV for test02/test03/test04 unless ctt_unknown_ablation.csv is later generated.
- FPR budgets 5e-3 and 1e-2 are not in the original low-FPR CSV; they were recomputed only for completed ROAD and CT&T test01 score dumps.
- Failure-case analysis for CT&T shifted settings is incomplete until all CT&T prediction dumps exist.
- To补齐: run `python -m cmf_can.training.cli --dataset <dataset> --model <model> --eval-only --save-predictions [--save-gate-weights] --num-workers 0`.
- Recommended saved fields: sample_id, dataset, setting, model, label, prediction, score, attack_type, vehicle, window_start, window_end, split, gate_frame, gate_window, gate_context, embedding_path.
