# Missing or Failed Runs

- P0 ROAD full matrix not completed: basic, time-bias, same-ID+time-bias, attention-pool, top-k, TFS concat, TFS stats-token attention.
- CT&T test01/test02 transformer-rescue candidates were not trained because ROAD did not exceed Transformer F1.
- Focal loss + val_aupr and weighted CE + val_recall_at_fpr_1e-3 training were not run for rescue candidates.
- Validation prediction dumps were not exported, so precision-constrained validation threshold is unavailable.
- True Mamba/SSM baseline was not run; no mamba/causal-conv/selective-scan dependency is available in the current environment.
