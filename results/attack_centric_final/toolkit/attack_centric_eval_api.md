# Attack-Centric Evaluation API

`cmf_can.analysis.attack_centric_eval` provides:
- `compute_binary_metrics(y_true, y_pred, attack_label=1)` for attack/normal precision, recall, F1, macro-F1, weighted-F1, accuracy, balanced accuracy, MCC and confusion matrix.
- `compute_score_metrics(y_true, y_score)` for AUROC, AUPR and Recall/Precision/F1@FPR budgets 1e-4, 5e-4, 1e-3, 5e-3, 1e-2.
- `compute_trivial_baselines(y_true)` for all-normal, all-attack and random rare-attack baselines.
- `compute_ranking_inversion(metrics_df)` for rank disagreement, Spearman/Kendall and top-k overlap.
- `compute_event_metrics(prediction_df)` for approximate event recall, delay and false alarms.

In single-label classification, weighted recall equals accuracy because each class recall is weighted by class support, so the numerator becomes total correct predictions. In rare-attack IDS, this can hide a zero attack-F1 detector. Therefore papers must report attack-positive precision/recall/F1, AUPR, low-FPR recall and event-level evidence before claiming an IDS is effective.
