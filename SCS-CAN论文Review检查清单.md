# SCS-CAN 论文 Review 检查清单与结果评估文档

版本：v1.0  
用途：用于投稿前 review，检查当前实验、图表、创新点和论文叙事是否已经达到会议投稿要求。

## 0. 当前已有图表与结果文件

你目前已有的图表和表格已经比较完整：

- `fig1_main_results.pdf / .png`：主实验结果
- `fig2_few_label.pdf / .png`：少标签实验
- `fig3_ablation.pdf / .png`：消融实验
- `fig4_cross_vehicle.pdf / .png`：跨车实验
- `fig5_convergence.pdf / .png`：收敛曲线
- `fig6_precision_recall.pdf / .png`：Precision-Recall 分析
- `fig7_fpr.pdf / .png`：FPR 分析
- `main_results.csv`
- `ablation_results.csv`
- `few_label_results.csv`
- `cross_vehicle_results.csv`
- `scs_can_dev.json`
- `table1_main.tex`
- `table2_ablation.tex`
- `table3_fewlabel.tex`
- `table4_crossvehicle.tex`
- `table5_datasets.tex`
- `table6_detailed.tex`

这套图表已经覆盖一篇会议论文常见的主要实验模块：主结果、消融、少标签、跨车、收敛、误报率和 PR 分析。现在 review 的重点不是继续加图，而是检查这些结果是否能支撑论文创新点。

---

## 1. 论文主线是否成立

当前论文建议定位为：

> SCS-CAN：面向 CAN 总线入侵检测的多视角语义上下文增强方法。

不建议把论文写成“提出一种新的 Transformer 模型”。更稳妥的主线是：

> 现有 CAN IDS 往往直接将 CAN 帧作为普通序列输入，忽略 CAN 报文内部结构、Payload 字节语义、时间周期和 CAN ID 转移上下文。本文提出 SCS-CAN，从 CAN ID、Payload、时间间隔和 ID 转移上下文四个视角建模 CAN 报文语义，并通过 Transformer 编码窗口级通信上下文，从而提升 CAN 入侵检测效果。

论文主线成立需要满足：

| 检查项                        | 是否必须 | 达标标准                                       |
| -------------------------- | ----:| ------------------------------------------ |
| SCS-CAN 明显优于普通 Transformer | 必须   | ROAD / CrySyS-subset 至少一个主数据集明显提升          |
| 多视角模块有效                    | 必须   | w/o Transition Context 或 w/o Payload 后性能下降 |
| 实验协议可信                     | 必须   | split 合理、无窗口泄漏、pretrain normal-only        |
| 少标签有可解释结果                  | 强烈建议 | 低标签下 SCS-CAN 或 SCS-CAN+SSL 有优势             |
| 跨车实验存在                     | 强烈建议 | HCRL-SA 至少一组 Train A → Test B              |

---

## 2. 创新点 Review

### 2.1 创新点 1：CAN 报文多视角语义编码框架

建议写法：

> 本文提出一种面向 CAN 报文结构的多视角语义编码框架，从 CAN ID、Payload 字节、时间间隔和 ID 转移上下文四个角度分别提取语义表示，并融合为窗口级 CAN 通信表示。

这是当前论文的第一核心创新点。

需要由以下结果支撑：

- `fig1_main_results.pdf / .png`
- `main_results.csv`
- `table1_main.tex`
- `table6_detailed.tex`

Review 检查：

| 检查项                                       | 合格标准 |
| ----------------------------------------- | ---- |
| 是否包含 ROAD                                 | 必须   |
| 是否包含 CrySyS-subset                        | 强烈建议 |
| 是否包含 CNN / LSTM / Transformer baseline    | 必须   |
| SCS-CAN 是否明显优于 Transformer                | 必须   |
| SCS-CAN 是否明显优于 CNN / LSTM                 | 必须   |
| 是否同时报告 F1、Macro-F1、Precision、Recall、AUROC | 建议   |
| 是否有 FPR / FNR                             | 建议   |

如果结果满足 SCS-CAN > Transformer，可以写：

> 实验结果表明，SCS-CAN 在 ROAD 和 CrySyS-subset 上相较普通 Transformer 取得更优检测性能，说明仅依赖通用序列建模不足以充分捕获 CAN 报文语义，而针对 CAN 报文结构设计的多视角语义编码能够有效提升检测能力。

---

### 2.2 创新点 2：轻量级 CAN ID 转移上下文建模

建议写法：

> 本文设计轻量级 Transition Context Embedding，通过统计训练集中 CAN ID 的 top-k 后继转移关系，为每个 CAN ID 构造状态上下文表示，并将其注入序列模型中，以增强模型对 CAN 通信状态演化规律的感知能力。

需要由以下结果支撑：

- `fig3_ablation.pdf / .png`
- `ablation_results.csv`
- `table2_ablation.tex`

Review 检查：

| 检查项                                  | 合格标准 |
| ------------------------------------ | ---- |
| 是否有 w/o Transition Context           | 必须   |
| w/o Transition Context 是否下降          | 强烈建议 |
| transition graph 是否只用 train split 构建 | 必须   |
| val/test 是否未参与 transition 统计         | 必须   |
| top-k 是否说明清楚                         | 建议   |
| 是否强调轻量实现而非复杂 GNN                     | 建议   |

如果 w/o Transition 下降，可以写：

> 消融实验表明，移除 Transition Context 后模型性能下降，说明 CAN ID 的转移上下文能够为检测模型提供额外的状态语义信息，验证了 ID 转移上下文建模机制的有效性。

如果 w/o Transition 下降不明显，则写得保守一些：

> Transition Context 为模型提供了可解释的 CAN ID 邻域状态信息，并在部分攻击类型或少标签设置下起到辅助作用。

---

### 2.3 创新点 3：字段级语义一致性自监督任务

建议写法：

> 本文设计 Masked Field Modeling 和 ID-Payload Consistency 两类字段级自监督任务，用于学习正常 CAN 通信中的字段一致性关系，并分析其在少标签和阈值敏感场景中的作用。

注意：如果 Full SCS-CAN 在主性能上不如 w/o SSL，不要写“自监督显著提升整体检测性能”。

需要由以下结果支撑：

- `fig2_few_label.pdf / .png`
- `fig3_ablation.pdf / .png`
- `fig6_precision_recall.pdf / .png`
- `fig7_fpr.pdf / .png`
- `few_label_results.csv`
- `ablation_results.csv`
- `table3_fewlabel.tex`
- `table2_ablation.tex`

Review 检查：

| 检查项                            | 合格标准  |
| ------------------------------ | ----- |
| Full SCS-CAN 在少标签下是否优于 w/o SSL | 强烈建议  |
| w/o MFM 是否下降                   | 建议    |
| w/o IPC 是否下降                   | 建议    |
| SSL 是否提高 Precision 或降低 FPR     | 可作为补充 |
| 预训练是否只使用 normal windows        | 必须    |

写作策略：

- 如果少标签提升明显：SSL 可以作为第三创新点。
- 如果主性能不提升但 Precision/FPR 更好：写成 precision-recall trade-off 分析。
- 如果 SSL 全面不提升：把 SSL 降级为探索性模块，不要放在标题核心。

---

## 3. 数据集 Review

### 3.1 ROAD：主实验数据集

ROAD 应承担：

- 主性能实验
- 消融实验
- 少标签实验

必须检查：

| 检查项                      | 合格标准                  |
| ------------------------ | --------------------- |
| split 是否修正               | train/val/test 攻击比例合理 |
| 是否输出 split_stats.csv     | 必须                    |
| 是否无窗口跨 split             | 必须                    |
| 是否无重叠泄漏                  | 必须                    |
| SCS-CAN 是否优于 Transformer | 必须                    |
| 是否有消融                    | 必须                    |

危险情况：

```text
train attack ratio = 2%
val/test attack ratio = 40%+
```

这种 split 下的结果不能用于论文主结论。

---

### 3.2 CrySyS-subset：增强实验数据集

CrySyS-subset 很有价值，但要明确写是 subset。

必须检查：

| 检查项                          | 合格标准                   |
| ---------------------------- | ---------------------- |
| subset 大小是否写清楚               | CrySyS-5M / CrySyS-10M |
| 采样方式是否写清楚                    | 前 N 帧、按文件抽样或按攻击抽样      |
| 是否输出 split stats             | 必须                     |
| 是否避免窗口泄漏                     | 必须                     |
| SCS-CAN 是否优于或不弱于 Transformer | 强烈建议                   |

论文写法：

> Due to storage and computational constraints, we construct a CrySyS-subset containing X million frames for additional evaluation. The subset preserves temporal order and attack labels, and is split using the same leakage-free temporal protocol.

不要写成“在 CrySyS 全量数据集上验证”，除非你确实跑了全量。

---

### 3.3 HCRL-CH：传统攻击兼容性数据集

如果 HCRL-CH 上所有模型都接近 99% 或 100%，它不适合作为核心卖点。

它应承担：

- 传统 DoS / fuzzing / spoof 攻击兼容性验证

论文写法：

> 在 HCRL-CH 上，各模型均取得较高性能，说明该数据集中的传统攻击具有较明显统计特征。SCS-CAN 在该数据集上保持稳定性能，验证了其对传统注入类攻击的兼容性。

不要把 HCRL-CH 当成证明 SCS-CAN 创新性的核心数据集。

---

### 3.4 HCRL-SA：跨车泛化数据集

HCRL-SA 应承担：

- 跨车泛化实验

必须检查：

| 检查项                                    | 合格标准 |
| -------------------------------------- | ---- |
| vehicle 字段是否正确                         | 必须   |
| train/test 是否车辆隔离                      | 必须   |
| window 长度是否固定为 100                     | 必须   |
| 是否修复 collate bug                       | 必须   |
| 是否至少有 Train vehicle A → Test vehicle B | 必须   |
| SCS-CAN 是否优于 Transformer               | 强烈建议 |

如果 HCRL-SA 结果好，论文说服力会上一个台阶。

如果 HCRL-SA 结果一般，可以写：

> 跨车检测仍然具有挑战性，不同车辆之间 CAN ID 空间和 payload 分布存在明显差异。SCS-CAN 在该设置下保持了相对稳定的性能，说明其语义表示具有一定泛化能力。

---

## 4. 图表 Review

### 4.1 Figure 1：主实验结果

文件：

- `fig1_main_results.pdf`
- `fig1_main_results.png`
- `main_results.csv`
- `table1_main.tex`

该图应证明：

1. SCS-CAN 优于 CNN / LSTM / Transformer。
2. ROAD 与 CrySyS-subset 上有主性能支撑。
3. HCRL-CH 上保持传统攻击兼容性。

检查：

| 检查项               | 合格标准 |
| ----------------- | ---- |
| 图例清晰              | 必须   |
| 坐标轴清晰             | 必须   |
| 数据集名称明确           | 必须   |
| 指标名称明确            | 必须   |
| 图和 csv / tex 是否一致 | 必须   |

---

### 4.2 Figure 2：少标签实验

文件：

- `fig2_few_label.pdf`
- `fig2_few_label.png`
- `few_label_results.csv`
- `table3_fewlabel.tex`

该图应证明：

1. 少标签下 SCS-CAN 仍有效。
2. SSL 是否带来低标签收益。
3. 1%、5%、10% 是重点。

检查：

| 检查项                                               | 合格标准 |
| ------------------------------------------------- | ---- |
| 是否包含 1%、5%、10%、20%、100%                           | 必须   |
| 是否比较 Transformer / SCS-CAN w/o SSL / Full SCS-CAN | 必须   |
| 曲线是否清晰                                            | 必须   |
| 是否区分不同数据集                                         | 建议   |

如果 Full SCS-CAN 少标签表现好：

> Few-label results demonstrate that the field-level self-supervised objectives help SCS-CAN maintain stronger performance when labeled attack windows are scarce.

如果 Full 不好：

> Few-label results show that SCS-CAN’s CAN-specific multi-view representation remains effective under limited labels. The SSL-enhanced variant shows different precision-recall trade-offs.

---

### 4.3 Figure 3：消融实验

文件：

- `fig3_ablation.pdf`
- `fig3_ablation.png`
- `ablation_results.csv`
- `table2_ablation.tex`

这是证明创新点最关键的图。

应包含：

- Full SCS-CAN
- w/o SSL
- w/o MFM
- w/o IPC
- w/o Transition Context
- 最好有 w/o Payload 或 w/o Time

检查：

| 检查项                     | 合格标准 |
| ----------------------- | ---- |
| 每个消融项对应一个方法模块           | 必须   |
| w/o Transition 是否下降     | 强烈建议 |
| w/o IPC / w/o MFM 是否有影响 | 建议   |
| 指标是否统一                  | 必须   |
| 图和表是否一致                 | 必须   |

---

### 4.4 Figure 4：跨车实验

文件：

- `fig4_cross_vehicle.pdf`
- `fig4_cross_vehicle.png`
- `cross_vehicle_results.csv`
- `table4_crossvehicle.tex`

检查：

| 检查项                               | 合格标准 |
| --------------------------------- | ---- |
| 是否标明 Train vehicle / Test vehicle | 必须   |
| 是否车辆隔离                            | 必须   |
| 是否至少比较 Transformer 和 SCS-CAN      | 必须   |
| 是否有 HCRL-SA                       | 必须   |
| 是否有跨车结论                           | 必须   |

---

### 4.5 Figure 5：收敛曲线

文件：

- `fig5_convergence.pdf`
- `fig5_convergence.png`

检查：

| 检查项             | 合格标准 |
| --------------- | ---- |
| loss 是否下降       | 必须   |
| val metric 是否稳定 | 必须   |
| 是否存在明显过拟合       | 需要说明 |
| 是否能支撑训练稳定性      | 必须   |

论文写法：

> The convergence curves show that SCS-CAN can be trained stably under the adopted optimization setting.

---

### 4.6 Figure 6：Precision-Recall 分析

文件：

- `fig6_precision_recall.pdf`
- `fig6_precision_recall.png`

该图用于解释：

- Full SCS-CAN 是否更保守；
- SSL 是否提高 precision；
- threshold 是否影响 recall。

写法：

> Precision-recall analysis shows that the SSL-enhanced variant tends to produce more conservative predictions, achieving high precision while reducing recall under the selected threshold.

前提是结果确实支持。

---

### 4.7 Figure 7：FPR 分析

文件：

- `fig7_fpr.pdf`
- `fig7_fpr.png`

该图用于说明误报率。

检查：

| 检查项             | 合格标准 |
| --------------- | ---- |
| FPR 是否清晰标注      | 必须   |
| 是否比较多个模型        | 必须   |
| 是否结合 Recall 解释  | 必须   |
| SCS-CAN 是否低 FPR | 建议   |

论文写法：

> Since false alarms are costly in in-vehicle IDS deployment, FPR is an important metric. SCS-CAN achieves a favorable false-positive profile compared with baseline models while maintaining competitive recall.

---

## 5. 表格 Review

### 5.1 Table 1：主实验表

文件：

- `table1_main.tex`
- `main_results.csv`

必须包含：

- Dataset
- Model
- Precision
- Recall
- F1
- Macro-F1
- AUROC
- FPR
- FNR

重点检查：

1. ROAD 是否有。
2. CrySyS-subset 是否有。
3. HCRL-CH 是否有。
4. SCS-CAN 是否优于 Transformer。
5. 是否和 Fig. 1 一致。

---

### 5.2 Table 2：消融实验表

文件：

- `table2_ablation.tex`
- `ablation_results.csv`

必须包含：

- Full SCS-CAN
- w/o SSL
- w/o MFM
- w/o IPC
- w/o Transition Context

重点检查：

1. 每个消融项是否对应论文模块。
2. 删除关键模块后是否下降。
3. 若不下降，正文是否解释。

---

### 5.3 Table 3：少标签实验表

文件：

- `table3_fewlabel.tex`
- `few_label_results.csv`

必须包含：

- 1%
- 5%
- 10%
- 20%
- 100%

重点检查：

1. Full SCS-CAN 在低标签下是否有效。
2. w/o SSL 与 Full 的区别是否合理。
3. Transformer baseline 是否从头训练。
4. Full 是否正确加载 SCS-CAN encoder 预训练权重。

---

### 5.4 Table 4：跨车实验表

文件：

- `table4_crossvehicle.tex`
- `cross_vehicle_results.csv`

必须包含：

- Train Vehicle
- Test Vehicle
- Model
- F1 / Macro-F1

重点检查：

1. 车辆是否隔离。
2. 是否至少一组完整跨车结果。
3. 是否与 Fig. 4 一致。

---

### 5.5 Table 5：数据集统计表

文件：

- `table5_datasets.tex`

建议包含：

- Dataset
- Frames
- Windows
- Attack Types
- Train Attack Ratio
- Val Attack Ratio
- Test Attack Ratio
- Usage

必须写清楚：

```text
CrySyS-subset, not full CrySyS
```

---

### 5.6 Table 6：详细结果表

文件：

- `table6_detailed.tex`

建议包含：

- Dataset
- Model
- Accuracy
- Precision
- Recall
- F1
- Macro-F1
- AUROC
- FPR
- FNR
- Threshold

它可以作为正文详细表或附录，用于支撑 Fig. 1、Fig. 6、Fig. 7。

---

## 6. 实验可信性硬检查

这些项目比模型结果更重要。如果这些有问题，结果再高也不能用。

### 6.1 split_stats.csv

每个数据集必须有：

```text
dataset, split, frames, windows, normal_windows, attack_windows, attack_ratio, attack_type_distribution
```

检查：

- train/val/test 攻击比例是否合理；
- 各 split 是否都有足够正常样本；
- test 是否包含目标攻击；
- 是否存在 train 几乎无攻击、test 大量攻击的情况。

---

### 6.2 window_index_check.log

必须检查：

```text
min_window_len = 100
max_window_len = 100
bad_windows = 0
cross_boundary_windows = 0
```

如果出现类似 `100 vs 117273`，该数据集结果作废。

---

### 6.3 transition_graph_stats.csv

必须检查：

```text
constructed_from_train_only = true
```

val/test 不能参与 transition graph 统计。

---

### 6.4 pretrain_normal_windows.log

必须检查：

```text
pretrain windows 均为 label=0 normal windows
attack_windows_excluded > 0
```

如果攻击窗口混入 normal pretraining，自监督结论不可靠。

---

### 6.5 少标签权重加载

必须确认：

```text
Transformer: from scratch
SCS-CAN w/o SSL: from scratch
SCS-CAN + SSL: load SCS-CAN pretrained encoder
```

不能把 SCS-CAN checkpoint 加载给 Transformer。

---

## 7. 是否达到投稿要求

### 7.1 最低可投稿标准

若目标是智能汽车安全相关会议，最低应满足：

| 项目       | 最低要求                                         |
| -------- | -------------------------------------------- |
| 主数据集     | ROAD 必须有                                     |
| 增强数据集    | CrySyS-subset 强烈建议有                          |
| 传统攻击数据集  | HCRL-CH 可作为补充                                |
| 跨车实验     | HCRL-SA 至少一组                                 |
| baseline | CNN、LSTM、Transformer                         |
| 主方法      | SCS-CAN 明显优于 Transformer                     |
| 消融       | w/o Transition Context 必须有                   |
| 少标签      | 1%、5%、10%、20%、100%                           |
| 图表       | main、few-label、ablation、cross-vehicle、PR、FPR |
| 泄漏检查     | split、window、transition、pretrain             |

---

### 7.2 较稳投稿标准

更稳的标准：

1. ROAD 上 SCS-CAN 明显优于 Transformer。
2. CrySyS-subset 上 SCS-CAN 明显优于或至少稳定优于 Transformer。
3. w/o Transition Context 明显下降。
4. 少标签 1%、5%、10% 下 SCS-CAN 或 SCS-CAN+SSL 优于 Transformer。
5. HCRL-SA 跨车实验中 SCS-CAN 优于 Transformer。
6. FPR 分析显示 SCS-CAN 误报率更低或可控。
7. 所有 split attack ratio 合理。
8. 所有窗口长度固定且无泄漏。
9. transition graph 只由 train 构建。
10. pretrain 只用 normal windows。

---

## 8. 论文写作中应避免的说法

不要写：

```text
首次联合建模 CAN ID 和 Payload。
```

不要写：

```text
首次使用 CAN ID 转移图。
```

不要写：

```text
自监督预训练显著提升所有检测性能。
```

不要写：

```text
在 CrySyS 上验证。
```

如果只用了 subset，应写：

```text
在 CrySyS-subset 上验证。
```

不要写：

```text
SCS-CAN 是一种全新的 Transformer 架构。
```

应写：

```text
SCS-CAN 是一种面向 CAN 报文结构的多视角语义上下文建模框架，采用 Transformer 作为序列编码器。
```

---

## 9. 建议最终贡献表述

建议写成四点：

```text
(1) We propose a CAN-specific multi-view semantic representation framework that jointly models CAN identifier semantics, payload byte patterns, timing intervals, and transition-context information.

(2) We design a lightweight CAN ID transition-context embedding mechanism that injects normal communication state information into a Transformer-based sequence encoder without using complex graph neural networks.

(3) We investigate field-level semantic consistency learning through Masked Field Modeling and ID-Payload Consistency objectives, and analyze their effects under full-label and few-label settings.

(4) We evaluate SCS-CAN on ROAD, CrySyS-subset, HCRL-CH, and HCRL-SA, covering standard detection, traditional injection attacks, few-label learning, ablation studies, and cross-vehicle generalization.
```

如果 SSL 效果不好，将第 3 点改成：

```text
We provide an analysis of field-level semantic consistency objectives...
```

而不是：

```text
We propose a self-supervised framework...
```

---

## 10. 摘要推荐写法

可参考：

```text
Controller Area Network (CAN) intrusion detection remains challenging because CAN frames contain limited explicit semantics, while attacks may manifest through subtle inconsistencies among identifiers, payload bytes, timing behavior, and message transition context. This paper proposes SCS-CAN, a CAN-specific multi-view semantic intrusion detection framework. SCS-CAN separately encodes CAN identifiers, payload bytes, timing intervals, and transition-context information, and fuses them into a Transformer-based sequence encoder for window-level detection. We further investigate field-level semantic consistency objectives, including Masked Field Modeling and ID-Payload Consistency, under full-label and few-label settings. Experiments on ROAD, CrySyS-subset, HCRL-CH, and HCRL-SA show that SCS-CAN improves detection performance over CNN, LSTM, and vanilla Transformer baselines, while maintaining competitive performance under few-label and cross-vehicle scenarios.
```

---

## 11. Introduction 推荐逻辑

建议按这个逻辑写：

1. CAN 缺少认证、加密和完整性校验。
2. DoS、Fuzzing、Spoof、Masquerade、Malfunction 等攻击威胁车载网络安全。
3. 现有深度学习 CAN IDS 常直接处理原始 CAN 序列，缺少对 CAN 报文结构的显式建模。
4. CAN 攻击可能破坏多个层面的语义一致性：ID-Payload、Payload 字节模式、时间周期、ID 转移上下文。
5. 因此，需要一种面向 CAN 结构的多视角语义建模方法。
6. 本文提出 SCS-CAN，并在 ROAD、CrySyS-subset、HCRL-CH、HCRL-SA 上验证。

---

## 12. Related Work 区分点

### 与 MIDS 的区别

```text
MIDS：ID/Payload 双流 + Bi-Mamba，强调监督式序列建模。
SCS-CAN：ID/Payload/Time/Transition 多视角融合，强调 CAN-specific semantic context。
```

### 与 Graph-based CAN IDS 的区别

```text
Graph-based 方法：构建 message sequence graph 或图特征。
SCS-CAN：使用轻量 top-k transition embedding 注入 Transformer，不引入复杂 GNN。
```

### 与 TrafficFormer 的区别

```text
TrafficFormer：通用网络流量预训练。
SCS-CAN：针对 CAN 字段结构设计 Masked Field Modeling 和 ID-Payload Consistency。
```

### 与普通 Transformer 的区别

```text
普通 Transformer：直接处理拼接后的 CAN 序列。
SCS-CAN：先进行 CAN-specific 多视角语义编码，再进行序列建模。
```

---

## 13. 最终 Review 结论模板

### 情况 A：全部达标

> 当前实验已经基本达到会议投稿要求。SCS-CAN 在 ROAD 和 CrySyS-subset 上相较普通 Transformer 取得明显提升，消融实验验证了 Transition Context 和多视角编码的有效性，少标签实验展示了方法在标注受限场景下的鲁棒性，HCRL-SA 跨车实验提供了泛化能力验证。建议进入论文写作阶段，重点突出 CAN-specific multi-view semantic encoding 和 lightweight transition context，将 SSL 作为辅助分析模块。

### 情况 B：主性能达标但 SSL 不达标

> 当前实验可以支撑 SCS-CAN 作为多视角语义上下文增强方法投稿，但不建议将自监督作为论文标题或第一贡献。应将 SSL 部分改为字段一致性分析或少标签增强模块，并在正文中客观说明其 precision-recall trade-off。

### 情况 C：ROAD 达标但 CrySyS 不达标

> 当前结果能证明方法在 ROAD 上有效，但跨数据集说服力不足。建议检查 CrySyS-subset 的 split、攻击比例和采样方式；若确认无误，可将 CrySyS 作为挑战性补充实验，并降低其在主文中的权重。

### 情况 D：消融不达标

> 如果 w/o Transition Context 或 w/o Payload Encoder 没有明显下降，则当前创新点需要重新收缩为工程性多视角融合，而不能强调 transition context 是核心贡献。建议增加更细粒度攻击类型分析，观察该模块是否在 spoof / masquerade 类攻击上更有效。

---

## 14. 最终需要逐项回答的 10 个问题

1. `main_results.csv` 中 ROAD 上 SCS-CAN 是否明显优于 Transformer？
2. `main_results.csv` 中 CrySyS-subset 上 SCS-CAN 是否优于或不弱于 Transformer？
3. `ablation_results.csv` 中 w/o Transition Context 是否下降？
4. `ablation_results.csv` 中 w/o IPC / w/o MFM 是否有影响？
5. `few_label_results.csv` 中 1%、5%、10% 标签下 SCS-CAN 是否优于 Transformer？
6. `cross_vehicle_results.csv` 中 HCRL-SA 是否 train/test 车辆隔离？
7. `split_stats.csv` 中 train/val/test 攻击比例是否合理？
8. 预训练是否只用了 normal windows？
9. transition graph 是否只用 train split 构建？
10. 所有窗口长度是否都是 100，没有再出现 `100 vs 117273` 的问题？

如果第 1、2、3、5、6、7、8、9、10 都通过，这篇论文已经具备投稿基础。

---

## 15. 总结

当前图表体系已经比较完整，包括：

```text
主实验
少标签实验
消融实验
跨车实验
收敛分析
Precision-Recall 分析
FPR 分析
数据集统计表
详细结果表
```

当前 review 的关键不是继续堆实验，而是确认：

```text
1. 结果是否真的支持创新点；
2. 图表是否与论文叙事一致；
3. 是否存在数据划分或窗口泄漏；
4. 是否避免夸大“首次”和“自监督提升”；
5. CrySyS 是否明确写为 subset。
```

如果这些都通过，当前实验已经可以进入论文写作阶段。
