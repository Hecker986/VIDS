#!/bin/bash
cd /root/autodl-tmp/scs-can
PY=/root/miniconda3/bin/python
LOG=logs/run_crysys.log
DS=crysys_subset

echo "=== CrySyS Subset Experiments ===" | tee $LOG
echo "Started: $(date)" | tee -a $LOG

echo "[1/6] Transformer" | tee -a $LOG
$PY src/training/train_baseline.py --root . --dataset $DS --model transformer --epochs 15 2>&1 | tee -a $LOG

echo "[2/6] CNN" | tee -a $LOG
$PY src/training/train_baseline.py --root . --dataset $DS --model cnn --epochs 15 2>&1 | tee -a $LOG

echo "[3/6] LSTM" | tee -a $LOG
$PY src/training/train_baseline.py --root . --dataset $DS --model lstm --epochs 15 2>&1 | tee -a $LOG

echo "[4/6] SCS-CAN w/o SSL" | tee -a $LOG
$PY src/training/finetune.py --root . --dataset $DS --model scs_can --variant wo_ssl --epochs 15 --ipc-weight 0 --table main_results.csv 2>&1 | tee -a $LOG

echo "[5/6] Pretrain" | tee -a $LOG
$PY src/training/pretrain.py --root . --dataset $DS --epochs 10 2>&1 | tee -a $LOG

echo "[6/6] SCS-CAN full" | tee -a $LOG
$PY src/training/finetune.py --root . --dataset $DS --model scs_can --variant full --epochs 15 --pretrained checkpoints/${DS}/pretrain/best.pt --ipc-weight 0.1 --table main_results.csv 2>&1 | tee -a $LOG

echo "" | tee -a $LOG
echo "=== ALL DONE ===" | tee -a $LOG
echo "Finished: $(date)" | tee -a $LOG
