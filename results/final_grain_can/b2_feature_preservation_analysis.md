# B2 Feature Preservation Analysis

The strongest audited fields are delta_t_same_id, payload statistics, payload bytes, can_id, and payload_delta_l1. Aggregate-window features preserve max/mean/std forms of these signals; the old deep window representation does not explicitly preserve them.
