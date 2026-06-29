# Final Model Decision

- Reliable-CMF-CAN Full should be abandoned as the main model.
- CAN-Transformer+ same-ID is the best completed rescue candidate, but it does **not** truly exceed Transformer F1.
- TFS-CAN gate should not be selected as main model.
- If the target is a stronger paper, the next necessary experiment is not more fusion; it is window-size/per-attack optimization or a real SSM/Mamba sequence backbone.
- Main metric should remain F1 for ROAD comparability, with AUPR/Recall@FPR reported as secondary deployment metrics.
