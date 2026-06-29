# CCF-A / Security Four Evidence Upgrade Report

This report adds security-reviewer-oriented evidence: event-level detection, false alarm rate per hour, open-world shifted settings, and normality-based unknown-attack repair. It does not claim that the current system already reaches CCF-A/security-four standards.

## What Was Added

- Attack event recall and detection delay.
- False alarm events per hour and false alarm windows per hour.
- Normality-based candidates fitted only on benign training windows for CT&T shifted settings.
- Open-world policy selection views: best window-F1, best event recall, and best event recall under <=10 false alarm events/hour.

## Best Open-World Rows

| Dataset | Selection | Method | F1 | Event recall | FA events/hour | Median delay (s) |
|---|---|---|---:|---:|---:|---:|
| ctt_test02 | best_window_f1 | normality:stats_robust_mean_rollmax9 / val_fpr_1em04 | 0.4689 | 0.6327 | 501.03 | 0.0 |
| ctt_test02 | best_event_recall | supervised:cmf_can / saved_val_threshold | 0.1600 | 1.0000 | 884.17 | 0.0 |
| ctt_test02 | best_event_recall_under_10_false_alarm_events_per_hour | normality:stats_ledoit_mahal_rollmean9 / val_fpr_1em04 | 0.0000 | 0.0000 | 0.00 | NA |
| ctt_test03 | best_window_f1 | normality:stats_ledoit_mahal / val_fpr_1em03 | 0.8570 | 0.9154 | 491.98 | 0.0 |
| ctt_test03 | best_event_recall | normality:stats_ledoit_mahal_rollmean9 / val_f1 | 0.8272 | 0.9775 | 1510.38 | 0.0 |
| ctt_test03 | best_event_recall_under_10_false_alarm_events_per_hour | normality:stats_ledoit_mahal / val_fpr_1em04 | 0.4157 | 0.1350 | 6.04 | 0.0 |
| ctt_test04 | best_window_f1 | normality:stats_robust_mean / val_f1 | 0.0984 | 0.1101 | 747.56 | 0.0 |
| ctt_test04 | best_event_recall | normality:stats_ledoit_mahal / val_f1 | 0.0225 | 1.0000 | 247.45 | 0.0 |
| ctt_test04 | best_event_recall_under_10_false_alarm_events_per_hour | normality:stats_robust_mean_rollmean9 / val_f1 | 0.0132 | 0.0088 | 8.34 | 0.0 |

## CCF-A / Security Four Gap Assessment

- Still insufficient for a top-tier claim if evaluated only by window-level F1: CT&T test04 remains weak.
- The evidence is stronger if the paper is reframed as open-world CAN IDS with event-level and low-false-alarm operating points.
- A top-tier submission still needs: adaptive-attacker evaluation, event-level metrics for the final anomaly ensemble per seed, self-supervised pretraining ablations, and cross-dataset transfer.

## Next Required Experiments

1. Export per-window scores for the full anomaly ensemble so event-level metrics can be computed for the final multi-seed policy, not only individual normality candidates.
2. Run context-masked CT&T test02/test04 with 3 seeds and record prediction dumps.
3. Add self-supervised pretraining and show improvement under 1% labels and unknown-attack shift.
4. Add adaptive attacker simulations: payload mimicry, ID replay, timing jitter, low-rate attacks.
5. Report false alarms per hour and detection delay in every main table.
