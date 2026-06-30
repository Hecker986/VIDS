# Final Reproduction Verdict

1. Target A original Bitbucket data obtained: yes.
2. Target B can-train-and-test-v1.5 data obtained: no.
3. test04≈0.998 reproduced in available sweeps: no.
4. If not reproduced, the current strongest available evidence is in `tables/exact_reproduction_sweep.csv`; Target B remains impossible to execute without the exact v1.5 source tree.
5. Existing local data can be used for Target A original Bitbucket alignment after the full public repository download, but not for Target B/can-sleuth v1.5 claims unless the v1.5 fingerprint matches.
6. Next step is to run full-negative/model-parameter exact sweeps on Target A and add Target B immediately if a v1.5 source is found or provided.
