# Missing Inputs Report

This report is generated from the current VIDS workspace and only records unavailable evidence. No unsupported figures were generated.

## Missing or Partial Inputs
- paper_fig7_recall_at_fpr: FPR budget 5e-3 unavailable in current CSVs; prediction scores are required.
- paper_fig7_recall_at_fpr: FPR budget 1e-2 unavailable in current CSVs; prediction scores are required.
- PR/ROC curves: not generated because it requires per-sample score/probability and label.
- Per-attack recall/F1: not generated because it requires per-sample prediction, label and attack_type.
- Gate weights by attack/setting: not generated because it requires saved gate_frame, gate_window and gate_context.
- UMAP/t-SNE representation: not generated because it requires embedding dump.
- Failure case analysis: not generated because it requires per-sample prediction.
- Calibration curve/reliability diagram: bin-level calibration data is missing; generated calibration summary table only.
- OOD score distribution: per-sample OOD scores are missing; generated summary bar/table only.

## Required Fields for Full Appendix Coverage
- sample_id
- dataset
- setting
- model
- label
- pred
- score
- attack_type
- vehicle
- gate_frame
- gate_window
- gate_context
- embedding_path

## Suggested Script Changes
- Add an evaluation export option in `cmf_can/training/cli.py` or `cmf_can/training/train.py` to save per-window predictions with the fields above.
- Add a CMF-CAN forward/evaluation hook in `cmf_can/models/cmf.py` / evaluation code to persist gate weights per sample.
- Add an embedding dump option before the classifier head for UMAP/t-SNE and failure-case analysis.
- Save calibration bin statistics if reliability diagrams are required.
