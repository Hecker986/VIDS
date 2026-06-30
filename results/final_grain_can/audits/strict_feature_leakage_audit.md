# Strict Feature Leakage Audit Revision

```csv
check,status,evidence
delta_t_same_id_past_only,pass,read_ctt_file updates last_ts before only after current row processing
payload_delta_l1_past_only,pass,read_ctt_file compares against previous same-ID payload
period_deviation_train_only,not_used,not included in final revision features
transition_profile_train_only,not_used,not included in final revision features
scaler_fit_train_only,pass,StandardScaler fit on train arrays only
can_id_no_test_label,pass,parsed from arbitration_id
negative_sampling_train_only,pass,collect_train reads train_01 only
threshold_validation_only,partial,model decision threshold from validation; low-FPR budget thresholds are upper bounds
attack_ratio_not_input,pass,"window_features returns attack_ratio separately, not concatenated into X"
timestamp_schedule_leakage,risk,timestamp-derived deltas can reflect capture schedule
file_identifier_feature,pass,file identifiers are not used as model features
episode_cross_split,risk,official CT&T split may contain related attack families across files
direct_label_proxy,pass,attack column excluded from features
large_negative_changes,partial,B 2x and C 5x partially completed; full-negative not completed
event_boundary_label_use,partial,event construction uses test labels for analysis only
duplicate_window_cross_split,not_checked,not fully audited at raw sample level
vehicle_file_leakage,pass,vehicle/file not used as features
```

Blocking direct leakage was not found. Remaining risks are timestamp/capture schedule, full-negative incompleteness, and approximate event boundaries.
