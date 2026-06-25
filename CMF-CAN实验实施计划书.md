# CMF-CAN 实验实施计划书

版本：v1.0  
目标：在本计划指导下完成数据下载、预处理、模型实现、实验评估、结果表格与论文素材整理，实验完成后可直接进入论文写作阶段。  
方法名称：**CMF-CAN: Cross-Modality Feature Fusion for Label-Efficient CAN Intrusion Detection**  
目标投稿：优先冲击较高水平网络安全 / 车联网安全 / 智能汽车安全方向会议；以 CCF B/C 或智能汽车安全相关会议作为保底；若实验结果充分，可进一步扩展为更高水平投稿版本。

---

## 0. 总体定位

### 0.1 一句话概括

CMF-CAN 是一种面向 CAN 总线入侵检测的跨模态特征融合方法。它将 CAN 流量表示为 **帧级序列模态、窗口级统计模态、ID 级上下文模态** 三类互补信息，并通过跨模态注意力和门控融合机制实现少标签、低误报、跨车辆和未知攻击场景下的鲁棒检测。

### 0.2 参考来源

本方案主要借鉴 CCS 2025 论文：

> **Training with Only 1.0‰ Samples: Malicious Traffic Detection via Cross-Modality Feature Fusion**

其核心思想是将网络流量组织为 packet / flow / host 多粒度多模态数据，并通过 cross-modality feature fusion 解决少样本恶意流量检测问题。  
官方源码：

```text
https://github.com/fuchuanpu/TFusion
```

本项目不直接沿用其网络流量数据处理流程，而是参考其跨模态融合设计，重新实现适配 CAN 数据结构的 PyTorch 版本。

### 0.3 核心问题

现有 CAN IDS 方法常见问题：

```text
1. 直接将 CAN 帧拼接成普通时序向量，忽略 CAN 报文本身的多模态结构。
2. 对少标签场景支持不足，攻击标签稀缺时模型性能下降明显。
3. 对跨车辆、未知攻击、低误报约束的评估不足。
4. 只依赖帧级序列，未充分利用窗口统计特征和 ID 级上下文特征。
```

CMF-CAN 的目标：

```text
1. 构造 CAN 版多粒度多模态表示。
2. 利用 cross-modal attention 建模不同模态之间的互补关系。
3. 主打 few-label CAN IDS。
4. 补充 low-FPR deployment-oriented evaluation。
5. 在 ROAD 和 CT&T 上形成有说服力的实验闭环。
```

---

## 1. 论文创新点

### 创新点 1：CAN 多粒度跨模态表示

本文将 CAN 数据组织为三类模态：

```text
1. Frame-level sequence modality
   逐帧 CAN ID、Payload、DLC、时间间隔、转移稀有度等序列特征。

2. Window-level statistical modality
   每个窗口的 ID 分布、Payload 分布、时间统计、转移统计等聚合特征。

3. ID-level context modality
   每个 CAN ID 在训练集中的周期、Payload profile、转移熵、出现频率等上下文画像。
```

与只输入原始帧序列不同，CMF-CAN 显式利用 CAN 通信的结构化语义。

### 创新点 2：CAN 版跨模态特征融合

本文设计 **Cross-Modality Fusion Block**，使帧级序列表示、窗口统计表示和 ID 上下文表示之间进行交互，而不是简单拼接。

核心思想：

```text
Frame branch 学习局部 CAN 报文序列模式；
Window-statistics branch 学习窗口级统计异常；
ID-context branch 学习 ID 级长期行为模式；
Cross-modal attention 学习三类模态之间的互补关系；
Gated fusion 动态分配不同模态的重要性。
```

### 创新点 3：面向少标签 CAN IDS 的训练与评估协议

本文重点评估少标签场景：

```text
1%, 5%, 10%, 20%, 100% labeled windows
```

目标是证明 CMF-CAN 在标注攻击样本有限时，仍能利用 CAN 结构信息获得更强检测能力。

### 创新点 4：面向实际部署的低误报评估

传统 IDS 只报告 F1 / Accuracy 不足以支撑车载部署。本文额外报告：

```text
Recall @ FPR ≤ 1e-4
Recall @ FPR ≤ 5e-4
Recall @ FPR ≤ 1e-3
F1 @ constrained FPR
FPR / FNR / Precision-Recall curve
```

目标是证明 CMF-CAN 不只提高平均指标，也能在严格误报约束下保持较高检测能力。

### 创新点 5：跨车辆 / 未知攻击泛化实验

依托 CT&T 数据集的标准组织方式，评估：

```text
known vehicle + known attack
unknown vehicle + known attack
known vehicle + unknown attack
unknown vehicle + unknown attack
```

该实验是论文的重要亮点之一，用于证明模型不只是单数据集刷分，而是在泛化协议下具有实用价值。

---

## 2. 数据集选择与下载链接

### 2.1 必做数据集 1：ROAD

**官方链接：**

```text
https://zenodo.org/records/10462796
```

**下载内容：**

```text
road.zip
```

**用途：**

```text
1. 主性能实验；
2. 少标签实验；
3. 低误报实验；
4. 消融实验；
5. 模型收敛与效率测试。
```

**特点：**

```text
1. 包含 ambient、fuzzing、fabrication、advanced attacks、simulated masquerade attacks；
2. 真实车辆动态台架采集；
3. 适合作为 CAN IDS 主数据集；
4. 数据规模适中，适合作为 P0 实验数据。
```

**注意事项：**

```text
1. 使用 leakage-aware time-block split；
2. 禁止随机窗口切分；
3. 必须输出 split_stats.csv；
4. 必须确保窗口不跨 split 边界；
5. 所有统计特征只能由 train split 计算。
```

### 2.2 必做数据集 2：CT&T / can-train-and-test

**官方链接：**

```text
https://data.dtu.dk/articles/dataset/can-train-and-test/24805533
```

**论文链接：**

```text
https://arxiv.org/abs/2308.04972
```

**数据特点：**

```text
1. 包含 4 辆车：
   - 2017 Subaru Forester
   - 2016 Chevrolet Silverado
   - 2011 Chevrolet Traverse
   - 2011 Chevrolet Impala

2. 提供 train_01 和四类 test：
   - test_01_known_vehicle_known_attack
   - test_02_unknown_vehicle_known_attack
   - test_03_known_vehicle_unknown_attack
   - test_04_unknown_vehicle_unknown_attack

3. CSV 格式，所有样本带标签；
4. 适合作为泛化实验核心数据集。
```

**下载建议：**

```bash
mkdir -p data/raw/ctt
cd data/raw/ctt
# 浏览器或 wget/curl 下载 data.dtu.dk 页面中的压缩包
# 下载后解压
unzip can-train-and-test.zip
```

### 2.3 可选增强数据集 3：CrySyS / CrySyS-subset

**论文链接：**

```text
https://www.nature.com/articles/s41597-023-02716-9
```

**用途：**

```text
1. 真实攻击补充验证；
2. 大规模 CAN 数据补充；
3. 如果资源不足，则只做 CrySyS-subset。
```

**使用策略：**

```text
1. 若完整数据处理和训练可承受，则报告 CrySyS；
2. 若只采样，则必须命名为 CrySyS-subset；
3. CrySyS-subset 需要保存 subset_manifest.json；
4. 不能把 subset 结果写成 full CrySyS 结果。
```

### 2.4 辅助数据集 4：HCRL-CH / OTIDS / Car-Hacking

**HCRL / OTIDS 下载链接：**

```text
https://ocslab.hksecurity.net/Dataset/CAN-intrusion-dataset
```

**用途：**

```text
1. 传统 DoS / fuzzing / spoofing sanity check；
2. 不作为论文核心数据集；
3. 如果所有模型接近满分，只作为兼容性实验。
```

---

## 3. 目录结构

建议创建新项目：

```text
cmf-can/
  README.md
  requirements.txt

  configs/
    road.yaml
    ctt.yaml
    crysys_subset.yaml
    hcrl_ch.yaml
    model_cmf_can.yaml
    train.yaml
    few_label.yaml
    fpr_eval.yaml

  data/
    raw/
      road/
      ctt/
      crysys/
      hcrl_ch/
    processed/
      road/
      ctt/
      crysys_subset/
      hcrl_ch/
    splits/
    cache/

  src/
    data_parsers/
      parse_road.py
      parse_ctt.py
      parse_crysys.py
      parse_hcrl_ch.py

    preprocessing/
      build_vocab.py
      build_splits.py
      build_window_index.py
      compute_train_stats.py
      build_id_context.py
      build_transition_features.py
      build_window_statistics.py
      audit_processed_dataset.py

    datasets/
      cmf_can_dataset.py
      few_label_sampler.py
      collate.py

    models/
      cmf_can.py
      branches.py
      fusion.py
      baselines/
        cnn.py
        lstm.py
        gru.py
        transformer.py
        can_transformer.py
        mlp_stats.py

    training/
      train.py
      train_baseline.py
      evaluate.py
      evaluate_fpr_constrained.py

    experiments/
      run_main.py
      run_few_label.py
      run_ctt_generalization.py
      run_fpr_constrained.py
      run_ablation.py
      run_efficiency.py
      export_tables.py
      plot_results.py

    utils/
      seed.py
      metrics.py
      logger.py
      io.py
      memory.py

  scripts/
    download_links.md
    prepare_road.sh
    prepare_ctt.sh
    prepare_crysys_subset.sh
    run_main.sh
    run_few_label.sh
    run_ctt.sh
    run_ablation.sh
    run_all.sh
    clean_cache.sh

  third_party/
    TFusion/

  results/
    tables/
    figures/
    logs/
    predictions/

  checkpoints/
```

---

## 4. 第三方源码使用说明

### 4.1 tFusion 源码位置

```bash
mkdir -p third_party
git clone https://github.com/fuchuanpu/TFusion third_party/TFusion
```

### 4.2 使用方式

不建议直接修改其原始数据管线，因为原项目面向 packet / flow / host 网络流量，CAN 数据结构不同。

建议使用方式：

```text
1. 阅读 third_party/TFusion/model/ 下模型结构；
2. 阅读 main.py / train_contrast.py / data_contrast.py 中训练流程；
3. 参考其 cross-modality feature fusion 设计；
4. 在 src/models/cmf_can.py 中重新实现 CAN 版本。
```

### 4.3 原论文模块到 CAN 模块的映射

| tFusion 原始概念 | CAN 迁移概念 |
|---|---|
| packet-level feature | frame-level CAN sequence |
| flow-level feature | window-level CAN statistics |
| host-level feature | ID-level CAN context |
| cross-modal attention | branch-to-branch attention |
| lightweight few-shot detector | label-efficient CAN IDS classifier |

---

## 5. 统一数据格式

所有数据解析后统一为：

```text
timestamp,
can_id,
dlc,
data0,
data1,
data2,
data3,
data4,
data5,
data6,
data7,
label,
attack_type,
dataset,
vehicle,
capture_id,
split_group
```

后续预处理生成：

```text
delta_t_global
delta_t_same_id
period_zscore
payload_delta_l1
payload_delta_l2
payload_change_flag
transition_prob
transition_rarity
topk_successor_hit
id_context_features
window_statistics_features
```

---

## 6. 数据预处理流程

### 6.1 Step 1：解析原始数据

#### ROAD

输入：

```text
candump .log
```

输出：

```text
data/processed/road/frames.parquet
```

要求：

```text
1. 解析 timestamp；
2. 解析 CAN ID；
3. 解析 payload；
4. payload 不足 8 字节补 0；
5. 按攻击文件或标签文件生成 label / attack_type；
6. 保留 capture_id；
7. 按 timestamp 排序。
```

#### CT&T

输入：

```text
CSV
```

输出：

```text
data/processed/ctt/frames.parquet
```

要求：

```text
1. 保留原始 set 信息；
2. 保留 train_01 / test_01 / test_02 / test_03 / test_04；
3. 保留 vehicle；
4. 保留 attack_type；
5. 保留 label；
6. 不打乱官方 train/test 结构。
```

#### CrySyS-subset

输入：

```text
CrySyS logs
```

输出：

```text
data/processed/crysys_subset/frames.parquet
data/processed/crysys_subset/subset_manifest.json
```

要求：

```text
1. 支持 --max_frames；
2. 记录 subset 来源；
3. 记录采样方式；
4. 所有表格命名为 CrySyS-subset。
```

### 6.2 Step 2：构建 CAN ID vocabulary

```bash
python -m src.preprocessing.build_vocab --dataset road
python -m src.preprocessing.build_vocab --dataset ctt
```

输出：

```text
data/processed/{dataset}/id_vocab.json
```

规则：

```text
1. CAN ID 是类别 token，不作为连续数值；
2. train split 中未见 ID 在 test 中映射为 UNK_ID；
3. CT&T 每个 set 可使用 set 内 vocab；
4. 跨 set 实验可使用 train set vocab + UNK。
```

### 6.3 Step 3：构建 split

#### ROAD

使用 leakage-aware time-block split：

```text
train: 70%
val: 10%
test: 20%
```

要求：

```text
1. 每个 capture 内部按时间块切分；
2. split 边界插入 buffer = window_size；
3. 禁止窗口跨 split；
4. 输出 split_stats.csv。
```

#### CT&T

使用官方 split：

```text
train_01
test_01_known_vehicle_known_attack
test_02_unknown_vehicle_known_attack
test_03_known_vehicle_unknown_attack
test_04_unknown_vehicle_unknown_attack
```

要求：

```text
1. 不重新随机划分官方测试集；
2. 训练只用 train_01；
3. 四个测试集分别报告。
```

### 6.4 Step 4：构建窗口索引

默认：

```yaml
window_size: 100
stride: 100
```

输出：

```text
data/processed/{dataset}/windows_index.npy
results/tables/{dataset}_window_index_check.log
```

检查项：

```text
min_window_len = 100
max_window_len = 100
bad_windows = 0
cross_boundary_windows = 0
```

窗口标签：

```text
label = 1 if any frame in window is attack else 0
```

---

## 7. 特征工程设计

CMF-CAN 的关键是把 CAN 特征分成三类模态。

### 7.1 模态 A：Frame-level sequence features

每个窗口为 100 帧，每帧特征包括：

```text
can_id_idx
payload[8]
dlc
delta_t_global
delta_t_same_id
period_zscore
payload_delta_l1
payload_delta_l2
payload_change_flag
transition_prob
transition_rarity
topk_successor_hit
```

#### A1. CAN ID

```text
can_id_idx = id_vocab[can_id]
```

输入 ID embedding。

#### A2. Payload

原始：

```text
data0 ... data7
```

编码方式：

```text
byte embedding + payload CNN
```

附加 payload delta：

```text
payload_delta_l1 = L1 distance to previous payload of same CAN ID
payload_delta_l2 = L2 distance to previous payload of same CAN ID
payload_change_flag = 1 if payload changed from previous same-ID frame else 0
```

#### A3. Time

```text
delta_t_global = timestamp_t - timestamp_{t-1}
delta_t_same_id = timestamp_t - previous_timestamp_same_id
period_zscore = (delta_t_same_id - train_mean_period_id) / train_std_period_id
```

所有均值和方差只从 train split 计算。

#### A4. Transition

从 train split 构建：

```text
P(ID_{t+1} | ID_t)
transition_rarity = -log(P + eps)
topk_successor_hit = 1 if ID_{t+1} in top-k successors of ID_t else 0
```

### 7.2 模态 B：Window-level statistical features

对每个 100 帧窗口统计：

#### B1. ID statistics

```text
num_unique_ids
id_entropy
top1_id_frequency_ratio
top3_id_frequency_ratio
rare_id_ratio
id_repetition_rate
```

#### B2. Payload statistics

```text
payload_byte_mean[8]
payload_byte_std[8]
payload_entropy
payload_change_rate
same_id_payload_delta_l1_mean
same_id_payload_delta_l1_max
same_id_payload_delta_l2_mean
same_id_payload_delta_l2_max
```

#### B3. Time statistics

```text
delta_t_global_mean
delta_t_global_std
delta_t_global_max
delta_t_same_id_mean
delta_t_same_id_std
period_zscore_mean
period_zscore_max
```

#### B4. Transition statistics

```text
transition_rarity_mean
transition_rarity_max
rare_transition_ratio
topk_miss_ratio
```

输出：

```text
window_stats: FloatTensor[B, d_stats]
```

统计特征需要标准化：

```text
z = (x - train_mean) / train_std
```

### 7.3 模态 C：ID-level context features

对每个 CAN ID，在 train split 中统计：

```text
id_frequency
id_frequency_ratio
mean_period
std_period
payload_mean[8]
payload_std[8]
payload_change_rate
transition_out_degree
transition_entropy
rare_successor_ratio
```

每帧根据 can_id 查表得到：

```text
id_context_seq: [B, L, d_id_context]
```

然后可进一步池化为：

```text
id_context_window_repr
```

注意：

```text
ID context 只由 train split 构建。
测试集中未见 ID 使用 UNK context。
```

---

## 8. CMF-CAN 模型结构

### 8.1 总体架构

```text
Input CAN Window
  |
  |-- Frame-level Sequence Branch
  |-- Window-level Statistics Branch
  |-- ID-level Context Branch
  |
Cross-Modality Fusion Block
  |
Gated Fusion
  |
Classifier
```

### 8.2 Branch 1：Frame-level Sequence Branch

输入：

```python
can_id: [B, L]
payload: [B, L, 8]
frame_numeric: [B, L, d_frame_num]
```

结构：

```text
CAN ID Embedding
Payload Byte Embedding + CNN
Frame numeric MLP
Concat + Linear Projection
Temporal Encoder
```

推荐实现：

```text
ID Embedding dim = 128
Byte Embedding dim = 16
Payload CNN output dim = 128
Frame numeric dim = 64
Frame projection dim = 256
Temporal Encoder = TransformerEncoder or GRU
```

建议第一版使用 TransformerEncoder：

```yaml
d_model: 256
num_layers: 3
nhead: 4
dim_feedforward: 512
dropout: 0.1
```

输出：

```text
z_frame: [B, 256]
```

池化方式：

```text
CLS token 或 attention pooling
```

### 8.3 Branch 2：Window-level Statistics Branch

输入：

```text
window_stats: [B, d_stats]
```

结构：

```text
MLP(d_stats -> 256 -> 256)
LayerNorm
Dropout
```

输出：

```text
z_stats: [B, 256]
```

### 8.4 Branch 3：ID-level Context Branch

输入：

```text
id_context_seq: [B, L, d_id_context]
```

结构：

```text
MLP per frame
attention pooling / mean pooling
MLP
```

输出：

```text
z_ctx: [B, 256]
```

### 8.5 Cross-Modality Fusion Block

将三个 branch 的输出作为三个 modality tokens：

```python
tokens = stack([z_frame, z_stats, z_ctx], dim=1)
# tokens: [B, 3, 256]
```

通过 MultiheadAttention：

```python
tokens_fused = CrossModalTransformer(tokens)
```

推荐结构：

```yaml
fusion_layers: 2
fusion_heads: 4
fusion_dim: 256
dropout: 0.1
```

### 8.6 Gated Fusion

学习三种模态权重：

```python
gate = softmax(MLP(tokens_fused.reshape(B, -1)))
# gate: [B, 3]
z = gate_frame * token_frame + gate_stats * token_stats + gate_ctx * token_ctx
```

输出：

```text
z_fused: [B, 256]
```

### 8.7 Classifier

```text
MLP(256 -> 128 -> 2)
```

输出：

```text
normal / attack
```

损失：

```text
Weighted CrossEntropy 或 Focal Loss
```

建议默认：

```text
Weighted CrossEntropy
```

类别权重从 train split 计算。

---

## 9. Baseline 设计

### 9.1 必做 baseline

```text
1. CNN
2. LSTM
3. GRU
4. Transformer
5. Frame-only Transformer
6. Stats-only MLP
7. Concat-Fusion CAN model
8. CMF-CAN
```

说明：

```text
Frame-only Transformer:
  只使用 frame-level sequence。

Stats-only MLP:
  只使用 window-level statistics。

Concat-Fusion CAN model:
  使用 frame / stats / context 三类模态，但简单 concat，不做 cross-modal attention 和 gated fusion。

CMF-CAN:
  完整方法。
```

### 9.2 可选 baseline

```text
1. MIDS
2. CANet
3. MLP/RF/XGBoost on window statistics
4. Isolation Forest
5. One-Class SVM
```

### 9.3 论文引用与复现策略

#### MIDS

如果时间允许，复现官方代码或复现其核心思想。若无法复现，可以作为 related work 引用，不直接比较数值。

注意：

```text
不能直接拿 MIDS 论文中的 F1 与本实验结果硬比，因为数据划分协议不同。
```

可以在论文中写：

```text
Due to differences in split protocols and threat models, we do not directly compare raw scores reported in prior work. Instead, all implemented baselines are evaluated under the same preprocessing and split protocol.
```

#### tFusion

tFusion 是方法来源，不作为 CAN baseline 原样运行。CMF-CAN 是其 CAN 适配版本。

---

## 10. 训练方案

### 10.1 默认训练参数

```yaml
epochs: 50
batch_size: 256
optimizer: AdamW
lr: 1e-4
weight_decay: 1e-4
scheduler: cosine
warmup_epochs: 3
early_stop_patience: 10
amp: true
num_workers: 8
seeds: [42, 2024, 2026]
```

### 10.2 如果训练不收敛

检查：

```text
1. learning rate 是否过大；
2. 类别权重是否过强；
3. train/val split 是否分布异常；
4. window label 是否过于稀疏；
5. window_stats 是否标准化；
6. ID context 是否只由 train 构建。
```

### 10.3 收敛要求

每个主实验需要保存：

```text
train_loss
val_loss
val_f1
val_auc
val_fpr
best_epoch
```

输出：

```text
results/figures/convergence_{dataset}_{model}.png
```

论文里只放收敛正常的曲线。

---

## 11. 实验设计

### 11.1 Experiment 1：主性能实验

数据集：

```text
ROAD
CT&T test_01_known_vehicle_known_attack
CrySyS-subset optional
HCRL-CH optional sanity check
```

模型：

```text
CNN
LSTM
GRU
Transformer
Frame-only Transformer
Stats-only MLP
Concat-Fusion CAN
CMF-CAN
```

指标：

```text
Accuracy
Precision
Recall
F1
Macro-F1
AUROC
AUPR
FPR
FNR
```

输出：

```text
results/tables/main_results.csv
results/tables/table_main.tex
results/figures/fig_main_results.png
```

目标效果：

```text
1. CMF-CAN 在 ROAD 上优于 Transformer 和 Concat-Fusion；
2. CMF-CAN 在 CT&T known vehicle/known attack 上优于或不弱于 Transformer；
3. HCRL-CH 如接近满分，只作为 sanity check。
```

### 11.2 Experiment 2：Few-label 实验

数据集：

```text
ROAD
CT&T train_01
```

标签比例：

```text
1%, 5%, 10%, 20%, 100%
```

模型：

```text
Transformer
Concat-Fusion CAN
CMF-CAN
```

采样策略：

```text
1. 在 train split 中按标签比例采样 labeled windows；
2. 保持 normal/attack 比例；
3. 每个比例使用 seeds = [42, 2024, 2026]；
4. 报告 mean ± std。
```

输出：

```text
results/tables/few_label_results.csv
results/tables/table_few_label.tex
results/figures/fig_few_label.png
```

目标效果：

```text
1. CMF-CAN 在 1%、5%、10% 标签下明显优于 Transformer；
2. 若 full-label 提升有限，few-label 仍可作为论文核心卖点；
3. 报告 mean ± std，避免单 seed 偶然性。
```

### 11.3 Experiment 3：CT&T 泛化实验

使用 CT&T 官方设置：

```text
train_01
test_01_known_vehicle_known_attack
test_02_unknown_vehicle_known_attack
test_03_known_vehicle_unknown_attack
test_04_unknown_vehicle_unknown_attack
```

模型：

```text
Transformer
Concat-Fusion CAN
CMF-CAN
```

指标：

```text
F1
Macro-F1
AUROC
FPR
FNR
```

输出：

```text
results/tables/ctt_generalization_results.csv
results/tables/table_ctt_generalization.tex
results/figures/fig_ctt_generalization.png
```

目标效果：

```text
1. CMF-CAN 在 unknown vehicle known attack 上优于 Transformer；
2. CMF-CAN 在 known vehicle unknown attack 上优于 Transformer；
3. unknown vehicle unknown attack 可能较难，若绝对值低，重点看相对提升。
```

该实验是论文核心亮点之一。

### 11.4 Experiment 4：FPR-constrained 低误报实验

数据集：

```text
ROAD
CT&T test_01 / test_02 / test_03 / test_04
```

模型：

```text
Transformer
Concat-Fusion CAN
CMF-CAN
```

指标：

```text
Recall @ FPR ≤ 1e-4
Recall @ FPR ≤ 5e-4
Recall @ FPR ≤ 1e-3
F1 @ FPR ≤ 1e-4
F1 @ FPR ≤ 5e-4
F1 @ FPR ≤ 1e-3
```

输出：

```text
results/tables/fpr_constrained_results.csv
results/tables/table_fpr_constrained.tex
results/figures/fig_recall_at_fpr.png
results/figures/fig_precision_recall.png
```

目标效果：

```text
1. CMF-CAN 在严格 FPR 约束下 Recall 更高；
2. 若普通 F1 提升不大，低误报指标可作为论文重要卖点；
3. 结果必须基于 test set 分数扫描阈值计算。
```

### 11.5 Experiment 5：消融实验

数据集：

```text
ROAD
CT&T test_01
```

模型变体：

```text
CMF-CAN
w/o window statistics branch
w/o ID context branch
w/o cross-modal attention
w/o gated fusion
frame-only
stats-only
concat fusion
```

输出：

```text
results/tables/ablation_results.csv
results/tables/table_ablation.tex
results/figures/fig_ablation.png
```

目标效果：

```text
1. CMF-CAN > concat fusion；
2. CMF-CAN > frame-only；
3. 去掉 window statistics 或 ID context 后性能下降；
4. 去掉 cross-modal attention 后性能下降。
```

### 11.6 Experiment 6：效率实验

模型：

```text
Transformer
Concat-Fusion CAN
CMF-CAN
```

指标：

```text
Params
FLOPs
Latency per window, batch_size=1
Throughput windows/s, batch_size=256
GPU memory
```

输出：

```text
results/tables/efficiency_results.csv
results/tables/table_efficiency.tex
```

目标效果：

```text
1. CMF-CAN 的开销可接受；
2. 若比 Transformer 稍慢，需要说明性能/低误报/少标签收益；
3. 若 CMF-CAN 比大 Transformer 更轻，可作为效率亮点。
```

---

## 12. 数据审计与泄漏控制

必须实现：

```bash
python -m src.preprocessing.audit_processed_dataset --dataset road
python -m src.preprocessing.audit_processed_dataset --dataset ctt
```

输出：

```text
results/tables/split_stats.csv
results/tables/{dataset}_window_index_check.log
results/tables/feature_stats_audit.csv
results/tables/id_context_audit.csv
results/tables/transition_feature_audit.csv
```

必须满足：

```text
1. 所有窗口长度固定为 100；
2. bad_windows = 0；
3. cross_boundary_windows = 0；
4. ID context 只由 train split 构建；
5. period statistics 只由 train split 构建；
6. window statistics 标准化参数只由 train split 计算；
7. CT&T 官方 test 不参与任何训练统计；
8. CrySyS-subset 必须有 subset_manifest.json。
```

---

## 13. 结果表格规划

### Table 1：Dataset Statistics

```text
Dataset | Frames | Windows | Vehicles | Attack Types | Split Protocol | Usage
```

### Table 2：Main Detection Results

```text
Dataset | Model | Precision | Recall | F1 | Macro-F1 | AUROC | FPR | FNR
```

### Table 3：Few-label Results

```text
Dataset | Model | 1% | 5% | 10% | 20% | 100%
```

### Table 4：CT&T Generalization

```text
Test Setting | Transformer | Concat-Fusion | CMF-CAN
known vehicle + known attack
unknown vehicle + known attack
known vehicle + unknown attack
unknown vehicle + unknown attack
```

### Table 5：FPR-constrained Results

```text
Dataset | Model | Recall@1e-4 | Recall@5e-4 | Recall@1e-3 | F1@1e-4 | F1@5e-4 | F1@1e-3
```

### Table 6：Ablation Study

```text
Variant | ROAD F1 | CT&T F1 | ROAD FPR | CT&T FPR
```

### Table 7：Efficiency

```text
Model | Params | FLOPs | Latency | Throughput | GPU Memory
```

---

## 14. 图表规划

```text
Figure 1: CMF-CAN architecture
Figure 2: Main results on ROAD and CT&T
Figure 3: Few-label performance curve
Figure 4: CT&T known/unknown vehicle/attack generalization
Figure 5: Recall under FPR constraints
Figure 6: Ablation study
Figure 7: Precision-Recall curves
Figure 8: Efficiency comparison
```

---

## 15. 代码实现任务清单

### Task 1：下载并组织数据集

```text
ROAD
CT&T
CrySyS optional
HCRL-CH optional
```

### Task 2：实现 parser

```text
parse_road.py
parse_ctt.py
parse_crysys.py optional
parse_hcrl_ch.py optional
```

### Task 3：实现特征工程

```text
compute_train_stats.py
build_id_context.py
build_transition_features.py
build_window_statistics.py
```

### Task 4：实现 Dataset

```text
cmf_can_dataset.py
few_label_sampler.py
collate.py
```

Dataset 返回：

```python
{
  "frame_seq": ...,
  "window_stats": ...,
  "id_context_seq": ...,
  "label": ...,
  "attack_type": ...,
  "vehicle": ...,
  "capture_id": ...
}
```

### Task 5：实现模型

```text
FrameBranch
WindowStatsBranch
IDContextBranch
CrossModalFusionBlock
GatedFusion
CMFCAN
```

### Task 6：实现 baseline

```text
CNN
LSTM
GRU
Transformer
Frame-only Transformer
Stats-only MLP
Concat-Fusion CAN
```

### Task 7：实现训练和评估

```text
train.py
train_baseline.py
evaluate.py
evaluate_fpr_constrained.py
```

### Task 8：实现实验脚本

```text
run_main.py
run_few_label.py
run_ctt_generalization.py
run_fpr_constrained.py
run_ablation.py
run_efficiency.py
```

### Task 9：导出论文结果

```text
export_tables.py
plot_results.py
```

---

## 16. 推荐运行顺序

### Phase 1：ROAD 最小闭环

```bash
bash scripts/prepare_road.sh
python -m src.experiments.run_main --dataset road
python -m src.experiments.run_ablation --dataset road
python -m src.experiments.run_few_label --dataset road
python -m src.experiments.run_fpr_constrained --dataset road
```

交付：

```text
ROAD 主结果
ROAD few-label
ROAD FPR-constrained
ROAD ablation
```

### Phase 2：CT&T 泛化闭环

```bash
bash scripts/prepare_ctt.sh
python -m src.experiments.run_ctt_generalization
python -m src.experiments.run_few_label --dataset ctt
python -m src.experiments.run_fpr_constrained --dataset ctt
```

交付：

```text
CT&T known/unknown vehicle/attack 结果
CT&T few-label
CT&T FPR-constrained
```

### Phase 3：补充数据集

```bash
bash scripts/prepare_crysys_subset.sh
python -m src.experiments.run_main --dataset crysys_subset
```

交付：

```text
CrySyS-subset 补充实验
```

### Phase 4：最终导出

```bash
python -m src.experiments.export_tables
python -m src.experiments.plot_results
```

交付：

```text
results/tables/*.csv
results/tables/*.tex
results/figures/*.png
```

---

## 17. 论文成功判据

### 17.1 必须满足

```text
1. CMF-CAN 在 ROAD 上优于 Transformer 或 Concat-Fusion；
2. CMF-CAN 在 ROAD few-label 1%、5%、10% 下明显优于 Transformer；
3. CT&T 至少一个 unknown setting 中 CMF-CAN 优于 Transformer；
4. CMF-CAN > concat fusion，证明 cross-modality fusion 有效；
5. FPR-constrained 指标至少有一个阈值下 CMF-CAN 明显更好；
6. 数据审计全部通过。
```

### 17.2 若要冲更高水平

```text
1. CT&T unknown vehicle + unknown attack 明显优于 Transformer；
2. ROAD 和 CT&T 两个数据集 few-label 均明显提升；
3. FPR-constrained recall 明显提升；
4. ablation 强烈支持 cross-modal attention 和 gated fusion；
5. 效率可接受；
6. 所有结果 mean ± std。
```

### 17.3 如果结果不理想

可调整叙事：

```text
1. 若 full-label 提升不大，但 few-label 提升明显：
   主打 label-efficient CAN IDS。

2. 若 F1 提升不大，但 FPR-constrained 指标好：
   主打 low-false-positive deployment-oriented IDS。

3. 若 CT&T 泛化好：
   主打 generalizable CAN IDS。

4. 若只有 ROAD 好：
   不建议强投高水平会议，需要补 CT&T 或重新调整模型。
```

---

## 18. 论文写作大纲

### Abstract

突出：

```text
1. CAN IDS 标签稀缺；
2. 现有方法忽略多粒度跨模态结构；
3. CMF-CAN 构造 frame/window/ID-context 三模态；
4. cross-modality fusion；
5. ROAD + CT&T + few-label + unknown vehicle/attack + FPR-constrained 结果。
```

### Introduction

逻辑：

```text
1. CAN 安全问题；
2. IDS 是重要防御；
3. 现有模型直接序列建模不足；
4. CAN 流量天然多模态；
5. 标签稀缺与泛化困难；
6. 提出 CMF-CAN。
```

### Method

包含：

```text
1. Problem formulation；
2. CAN multi-modal representation；
3. Frame-level branch；
4. Window-statistics branch；
5. ID-context branch；
6. Cross-modality fusion；
7. Training objective。
```

### Experiments

包含：

```text
1. Datasets；
2. Baselines；
3. Implementation details；
4. Main results；
5. Few-label results；
6. CT&T generalization；
7. FPR-constrained evaluation；
8. Ablation；
9. Efficiency。
```

### Discussion

包含：

```text
1. Why cross-modality fusion helps；
2. Failure cases；
3. Limitations；
4. Deployment considerations。
```

---

## 19. Codex 总执行指令

```text
请根据本计划实现 CMF-CAN 项目。

核心要求：
1. 借鉴 TFusion 的 cross-modality feature fusion 思想，但重新实现 CAN 版 PyTorch 模型。
2. 数据集优先实现 ROAD 和 CT&T。
3. ROAD 使用 leakage-aware time-block split。
4. CT&T 使用官方 train/test 设置。
5. 构造三类模态：
   - frame-level sequence；
   - window-level statistics；
   - ID-level context。
6. 所有统计量只从 train split 计算。
7. 实现 CrossModalFusionBlock 和 GatedFusion。
8. 实现 CNN、LSTM、GRU、Transformer、Frame-only、Stats-only、Concat-Fusion、CMF-CAN。
9. 完成 main、few-label、CT&T generalization、FPR-constrained、ablation、efficiency 六类实验。
10. 所有结果导出 csv、tex、png。
11. 所有主实验至少支持 seeds=[42,2024,2026]。
12. checkpoint 只保存 best.pt 和 last.pt。
13. 输出数据审计文件，确保无窗口泄漏、无统计泄漏、无 test 参与特征构建。
14. 实验完成后，结果应能直接用于论文写作。
```

---

## 20. 最终交付物

```text
1. cmf-can 代码仓库
2. 数据下载说明
3. 数据预处理脚本
4. 特征工程脚本
5. CMF-CAN 模型实现
6. baseline 实现
7. 主实验结果表
8. few-label 结果表
9. CT&T 泛化结果表
10. FPR-constrained 结果表
11. 消融实验表
12. 效率实验表
13. 所有论文图
14. 所有 LaTeX 表格
15. README 复现实验命令
```
