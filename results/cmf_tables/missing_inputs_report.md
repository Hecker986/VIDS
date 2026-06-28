# Missing Inputs Report

No fake figures were generated.

## Missing or Partial Inputs

- Remaining methodological limitation: CT&T unknown-setting ablation is single-seed eval-only from CT&T test01 checkpoints; add multi-seed retraining for stronger top-tier claims.

## Completed Evidence Files
- CT&T test02-test04 prediction dumps for Transformer, Concat-Fusion and CMF-CAN exist.
- CT&T test02-test04 CMF-CAN gate weight dumps exist.
- CT&T unknown-setting ablation table exists.
- Shifted PR/ROC, failure-case, per-attack, calibration-bin and gate-weight figures are generated from real dumps.

## Reproduction Commands
- Prediction/gate dump: `python -m cmf_can.training.cli --dataset <dataset> --model <model> --eval-only --save-predictions [--save-gate-weights] --num-workers 0`.
- Embedding dump: `python -m cmf_can.analysis.export_embeddings --datasets road ctt_test02 ctt_test04 --model cmf_can`.
- Figure/table refresh: `python -m cmf_can.analysis.export_paper_refined_assets --root .`.

## Recommended Saved Fields
sample_id, dataset, setting, model, label, prediction, score, attack_type, vehicle, window_start, window_end, split, gate_frame, gate_window, gate_context, embedding_path.
