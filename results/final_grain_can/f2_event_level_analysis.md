# F2 Event-Level Revision

Event-level metrics were recomputed from aggregate-window100 score dumps. Official attack boundaries were not available, so events are approximate label-transition events. False alarms per hour use aggregate-window index as fallback time because true real-time timestamps are not preserved in the aggregate dump.

```csv
dataset,model,granularity,event_recall,event_precision_if_definable,mean_detection_delay_seconds,median_detection_delay_seconds,false_alarm_events_per_hour,false_alarm_samples_per_hour,recall_at_0.1_FA_per_hour,recall_at_1_FA_per_hour,recall_at_10_FA_per_hour,event_boundary_quality
ctt_test01,GradientBoosting,aggregate_window100,0.6666666666666666,,58.333333333333336,0.0,,0.1262670548209463,,,,approximate_from_labels
ctt_test02,GradientBoosting,aggregate_window100,0.357504215851602,,1048.5849056603774,0.0,,0.4466847615354788,,,,approximate_from_labels
ctt_test03,GradientBoosting,aggregate_window100,0.8906935908691835,,20.00985707244948,0.0,,0.1250767253060326,,,,approximate_from_labels
ctt_test04,GradientBoosting,aggregate_window100,0.3650107991360691,,42.60355029585799,0.0,,0.6263237518910741,,,,approximate_from_labels
```

Interpretation: CT&T test04 event recall is still limited, so the evidence does not support a strong unknown-attack deployment claim. This analysis is useful as a conservative appendix or limitation discussion.
