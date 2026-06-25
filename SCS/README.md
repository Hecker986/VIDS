# SCS-CAN 一周冲刺实验

协议版本：v3.0  
工作目录：`/root/autodl-tmp/scs-can`

## 数据集

- ROAD / HCRL-CH / HCRL-SA：主线
- CrySyS：可选（磁盘 <15GB 时跳过）

## 快速开始

```bash
cd /root/autodl-tmp/scs-can
export PYTHONPATH=/root/autodl-tmp/scs-can:$PYTHONPATH

# 预处理
bash scripts/prepare_dataset.sh hcrl_ch
bash scripts/prepare_dataset.sh road

# Day1 Transformer baseline
python src/training/train_baseline.py --dataset hcrl_ch --model transformer

# SCS-CAN 全流程
python src/training/pretrain.py --dataset hcrl_ch
python src/training/finetune.py --dataset hcrl_ch --variant full --pretrained checkpoints/hcrl_ch/pretrain/best.pt
```

## 结果

- `results/tables/main_results.csv`
- `results/tables/few_label_results.csv`
- `results/tables/cross_vehicle_results.csv`
- `results/tables/ablation_results.csv`
