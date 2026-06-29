# Feature and Leakage Audit

1. `delta_t_same_id` is computed from previous timestamp of the same CAN ID only.
2. `payload_delta_l1` is computed from previous payload of the same CAN ID only.
3. StandardScaler and robust normality median/MAD are fitted on train data only.
4. Transition/profile features beyond past deltas were not added in this run; no test-derived transition profile is used.
5. Test labels are used only for evaluation metrics, not feature construction or threshold selection.
6. Sample-level features use timestamp, arbitration_id, data_field-derived bytes/DLC, and past-only deltas; no direct label field is included.
7. Negative sampling affects training variance, so 5 seeds are reported. Full negative training was not completed because CT&T train_01 has over 10M frames.
8. Short-window `attack_ratio` is saved only as an audit column. It is not included in the model feature matrix; an earlier leaked run was discarded and the final `granularity_search.csv` was regenerated without `attack_ratio` as input.
