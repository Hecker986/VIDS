# Final Experiment Recommendation

1. Main model: keep CMF-CAN as the system-study model, but report simplified variants when they win.
2. Continue using Full CMF-CAN only for settings where it is competitive; do not force it as universal best.
3. Add threshold calibration / operating-point selection for deployment reporting.
4. Modality dropout is worth testing multi-seed, but existing evidence does not justify replacing the model globally.
5. Stats+Frame or -Ctx should be considered for unknown vehicle/unknown attack settings.
6. Low-FPR should be a main evaluation axis, not the only headline.
7. Write: ranking gains, low-FPR gains in selected settings, modality-transfer analysis.
8. Do not write: CMF-CAN consistently outperforms Transformer or solves unknown attack generalization.
9. If only 3 days remain: run multi-seed threshold calibration and write limitations honestly.
10. If 2 weeks remain: implement Stats+Frame/Context-masked CMF and multi-seed D7/D8/D9.
