# GRAIN Feature and Granularity Analysis

Strong signals come from causal timing and payload-preserving features such as delta_t_same_id, payload_delta_l1, payload_sum/std, bytes and CAN ID. Aggregate window_100 preserves these summaries, unlike old deep window pipelines that can dilute rare evidence.
