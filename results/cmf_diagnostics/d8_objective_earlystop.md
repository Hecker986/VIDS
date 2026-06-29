# D8 Objective and Early Stopping

This run consolidates existing objective/selection experiments instead of launching another long training sweep. Evidence suggests low-FPR deployment should not rely only on val F1; selection_metric=AUPR or Recall@FPR should be tested in a follow-up multi-seed run.
