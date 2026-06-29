# Final Model Selection

Reliable-CMF-CAN Full should **not** replace CMF-CAN as the main model based on the completed runs.

- **ROAD**: Transformer or old CMF-CAN depending metric — Reliable-CMF-CAN F1=0.7877, lower than Transformer 0.8279 and old CMF-CAN 0.7894; Recall@FPR thresholding improves but not enough.
- **CT&T test02**: CMF-CAN -Ctx / anomaly calibrated policy — Reliable-CMF-CAN F1=0.1600 and Recall@FPR<=1e-3=0.4540; -Ctx/anomaly evidence is stronger.
- **CT&T test03/test04**: Anomaly/normality policy until Reliable is trained — No Reliable-CMF-CAN 50ep result in this run; existing anomaly policy is the only strong unknown-attack evidence.

Current result is CCF B-style evidence at best, not sufficient for CCF A / Security Four.
