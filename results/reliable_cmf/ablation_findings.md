# Ablation Findings

1. Reliability Gate 是否比普通 gate 更好：未能证明。Reliable-CMF-CAN Full 在已完成的 ROAD/test02 上没有超过旧 CMF-CAN 或最强简化变体。
2. Shift-aware Context 是否改善 test02/test04：未能证明。test02 Reliable Full F1 仍为 0.1600，明显低于已存在的 -Ctx 单 seed 结果。
3. Normality Branch 是否改善 unknown attack：Reliable 模型训练未包含可学习 normality 分支；现有 anomaly policy 仍是 unknown attack 的更强证据。
4. Segment Pooling 是否改善 ROAD/per-attack：未能证明。ROAD Reliable F1 低于 Transformer 和旧 CMF-CAN。
5. Full 是否仍输给简化模型：是。ROAD 上低于 Transformer/旧 CMF-CAN；CT&T test02 上低于 -Ctx 和 anomaly policy。
6. 当前主模型建议：不要切换到 Reliable-CMF-CAN Full；ROAD 仍用 Transformer/旧 CMF-CAN 做强 baseline，CT&T test02 用 -Ctx 或 anomaly/calibrated policy。
