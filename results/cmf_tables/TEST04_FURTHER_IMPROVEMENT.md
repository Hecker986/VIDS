# CT&T Test04 Further Improvement Attempt

Date: 2026-06-28

## Goal

Try stronger but still defensible assumptions to improve CT&T test04 beyond the current temporal CMF-CAN+Anomaly result.

Current final test04 multi-seed result:

| Metric | Baseline CMF-CAN | Temporal CMF-CAN+Anomaly |
|---|---:|---:|
| F1 mean | 0.0284 | 0.4457 |
| AUPR mean | 0.2210 | 0.2992 |
| Recall@FPR<=1e-4 mean | 0.0148 | 0.1329 |

## Additional Stronger Assumptions Tried

### 1. Target-domain benign calibration

Assumption:

- Before deployment, the target vehicle can provide benign calibration traffic.
- Thresholds are selected using held-out benign target windows.
- No target attack labels are used for threshold selection.

Result on seed 42:

- Best window-level F1 remained about 0.44.
- This did not improve over the temporal validation-threshold policy.

Interpretation:

- The failure is not only a threshold scale problem.
- Target benign calibration alone cannot create enough separation for unknown vehicle + unknown attack.

### 2. Event/alarm-level evaluation

Assumption:

- IDS decisions can be aggregated into alarm segments rather than evaluated as isolated windows.

Finding:

- CT&T test04 test split contains 463 positive event fragments, with median event length of 1 window.
- Direct event-level precision is poor because many alarms do not overlap the fragmented positive windows.

Interpretation:

- Event-level evaluation is not automatically easier on this prepared window index.
- To use event-level evaluation fairly, the dataset preparation would need attack-session metadata or a defensible event merge tolerance.

## Conclusion

The current best defensible result is the temporal CMF-CAN+Anomaly result:

- F1 mean: 0.4457
- AUPR mean: 0.2992
- Recall@FPR<=1e-4 mean: 0.1329

Pushing CT&T test04 to F1 >= 0.8 is unlikely under the current strict setting:

- window-level labels,
- no target attack labels,
- unknown vehicle,
- unknown attack,
- no attack-session event metadata.

## What Would Be Required for F1 >= 0.8

Getting F1 >= 0.8 would require changing the experimental assumption, not just tuning the existing detector:

1. **Target-labeled calibration**: use a small number of labeled target attack windows to tune thresholds or fusion weights.
2. **Event-session evaluation**: evaluate attack-session detection instead of per-window detection, with documented attack start/end metadata.
3. **Target-domain self-supervised adaptation**: retrain/adapt encoders on target vehicle traffic before evaluation.
4. **Full target normal baseline**: fit normality models from clean target-vehicle benign traces instead of source-domain normal traces.

These are valid future settings, but they should be clearly labeled as stronger-assumption experiments.
