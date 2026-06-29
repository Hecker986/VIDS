# Root Cause Ranking

1. **threshold/calibration mismatch** — AUROC/AUPR often stronger than F1; D1 threshold sweeps. Action: Use calibrated thresholds and report low-FPR operating points.
2. **modality over-fusion** — Full is beaten by -Ctx/Stats/-Gate in shifted ablation. Action: Use modality dropout/masking or present simplified variants.
3. **ID-context vehicle shift** — -Ctx improves test02/test04; D3 shows ID/frequency shift. Action: Mask/downweight context for low-overlap vehicles.
4. **window statistics robustness** — Stats-only strong in shifted low-FPR; D4 feature AUC. Action: Strengthen stats branch or Stats+Frame.
5. **attack-type heterogeneity** — Fuzzing/interval/systematic low recall. Action: Attack-aware calibration or targeted augmentation.
6. **window label dilution** — Low attack-frame-ratio buckets can have low recall. Action: Try window_size=50 or segment/top-k pooling.
7. **training objective mismatch** — Val F1 not aligned with low-FPR. Action: Early stop on AUPR/Recall@FPR.
8. **seed variance** — Few-label std is nontrivial. Action: Multi-seed unknown ablations.
9. **model capacity/regularization mismatch** — Robust variants not globally dominant. Action: Tune regularization after modality fixes.
10. **baseline too strong / task sequence-driven** — Transformer wins ROAD F1. Action: Keep Transformer as strong baseline.
