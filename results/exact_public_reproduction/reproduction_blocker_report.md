# Reproduction Blocker Report

Exact reproduction did not reach test04 attack-positive F1 >= 0.95.

1. Target A original Bitbucket data tree present: yes.
2. Target B v1.5 data version aligned: no.
3. Official feature table / can-sleuth preprocessing code: not present locally.
4. Current Target A sweep rows are marked `completed_sampling_approximation` unless a full-negative protocol is explicitly present.
5. Required next input for Target B: the exact can-train-and-test-v1.5 release or can-sleuth feature table/code path under `CTT_V15_ROOT`.

This file is an execution blocker only for reproducing the public high score, not a reason to stop downloading or auditing data.
