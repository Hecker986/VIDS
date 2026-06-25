#!/bin/bash
set -e
cd /root/autodl-tmp/scs-can
PY=/root/miniconda3/bin/python
LOG=logs/run_remaining2.log

echo "=== Remaining2 ===" | tee $LOG
echo "Started: $(date)" | tee -a $LOG

# Cross-vehicle (hcrl_sa transition graph should be ready)
echo "=== Cross-vehicle ===" | tee -a $LOG
$PY src/experiments/run_cross_vehicle.py --root . --train-vehicle kia --test-vehicle sonata 2>&1 | tee -a $LOG || echo "cross-vehicle failed, continuing" | tee -a $LOG

# CrySyS remaining (skip Transformer if already in main_results.csv)
if ! grep -q "crysys,cnn" results/tables/main_results.csv 2>/dev/null; then
  echo "[CrySyS] CNN" | tee -a $LOG
  $PY src/training/train_baseline.py --root . --dataset crysys --model cnn --epochs 15 2>&1 | tee -a $LOG
fi

if ! grep -q "crysys,lstm" results/tables/main_results.csv 2>/dev/null; then
  echo "[CrySyS] LSTM" | tee -a $LOG
  $PY src/training/train_baseline.py --root . --dataset crysys --model lstm --epochs 15 2>&1 | tee -a $LOG
fi

if ! grep -q "crysys,scs_can,wo_ssl" results/tables/main_results.csv 2>/dev/null; then
  echo "[CrySyS] SCS-CAN w/o SSL" | tee -a $LOG
  $PY src/training/finetune.py --root . --dataset crysys --model scs_can --variant wo_ssl --epochs 15 --ipc-weight 0 --table main_results.csv 2>&1 | tee -a $LOG
fi

if ! ls checkpoints/crysys/pretrain/best.pt 2>/dev/null; then
  echo "[CrySyS] Pretrain" | tee -a $LOG
  $PY src/training/pretrain.py --root . --dataset crysys --epochs 10 2>&1 | tee -a $LOG
fi

if ! grep -q "crysys,scs_can,full" results/tables/main_results.csv 2>/dev/null; then
  echo "[CrySyS] SCS-CAN full" | tee -a $LOG
  $PY src/training/finetune.py --root . --dataset crysys --model scs_can --variant full --epochs 15 --pretrained checkpoints/crysys/pretrain/best.pt --ipc-weight 0.1 --table main_results.csv 2>&1 | tee -a $LOG
fi

echo "" | tee -a $LOG
echo "=== ALL REMAINING2 DONE ===" | tee -a $LOG
echo "Finished: $(date)" | tee -a $LOG
