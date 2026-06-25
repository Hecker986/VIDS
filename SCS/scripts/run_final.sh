#!/bin/bash
cd /root/autodl-tmp/scs-can
PY=/root/miniconda3/bin/python
LOG=logs/run_final.log

echo "=== FINAL EXPERIMENTS ===" | tee $LOG
echo "Started: $(date)" | tee -a $LOG

# 1. Cross-vehicle
echo "" | tee -a $LOG
echo "=== Cross-vehicle ===" | tee -a $LOG
$PY src/experiments/run_cross_vehicle.py --root . --train-vehicle kia --test-vehicle sonata 2>&1 | tee -a $LOG || echo "[WARN] cross-vehicle failed" | tee -a $LOG

# 2. CrySyS subset - all 5 models
DS=crysys_subset

echo "" | tee -a $LOG
echo "=== CrySyS Subset ===" | tee -a $LOG

echo "[CrySyS] Transformer" | tee -a $LOG
$PY src/training/train_baseline.py --root . --dataset $DS --model transformer --epochs 15 2>&1 | tee -a $LOG

echo "[CrySyS] CNN" | tee -a $LOG
$PY src/training/train_baseline.py --root . --dataset $DS --model cnn --epochs 15 2>&1 | tee -a $LOG

echo "[CrySyS] LSTM" | tee -a $LOG
$PY src/training/train_baseline.py --root . --dataset $DS --model lstm --epochs 15 2>&1 | tee -a $LOG

echo "[CrySyS] SCS-CAN w/o SSL" | tee -a $LOG
$PY src/training/finetune.py --root . --dataset $DS --model scs_can --variant wo_ssl --epochs 15 --ipc-weight 0 --table main_results.csv 2>&1 | tee -a $LOG

echo "[CrySyS] Pretrain" | tee -a $LOG
$PY src/training/pretrain.py --root . --dataset $DS --epochs 10 2>&1 | tee -a $LOG

echo "[CrySyS] SCS-CAN full" | tee -a $LOG
$PY src/training/finetune.py --root . --dataset $DS --model scs_can --variant full --epochs 15 --pretrained checkpoints/${DS}/pretrain/best.pt --ipc-weight 0.1 --table main_results.csv 2>&1 | tee -a $LOG

echo "" | tee -a $LOG
echo "=== ALL FINAL DONE ===" | tee -a $LOG
echo "Finished: $(date)" | tee -a $LOG
