# ROAD Failure Analysis

Best completed rescue model by F1 is **Transformer** with F1=0.8279. The Transformer baseline remains higher at F1=0.8279.

1. Transformer 是否接近当前特征和数据协议上限：当前证据支持这个判断。CAN-specific same-ID features lifted AUPR slightly, but did not pass Transformer F1.
2. 无效模块：TFS residual gate did not help ROAD F1 or AUPR; window stats appear to dilute rather than improve the strong sequence signal.
3. 是否需要改 window size：需要。当前 window=100 可能让 sparse attack recall 受限；应优先测试 50/100/200。
4. 是否需要 Mamba/SSM：可以作为下一步，但必须使用真实 Mamba/SSM implementation，不应伪造 Mamba baseline。
5. 是否需要 per-attack 专门优化：需要，尤其是 sparse/fuzzing/short-span attacks。
6. 是否停止当前方案：应停止继续堆 fusion；保留 CAN-Transformer+ same-ID 作为 AUPR 改进证据，但不要称其稳定超过 Transformer。
