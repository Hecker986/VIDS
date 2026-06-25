#!/bin/bash
set -e
cd /root/autodl-tmp/scs-can
PY=/root/miniconda3/bin/python
LOG=logs/run_remaining.log

echo "=== Remaining Experiments ===" | tee $LOG
echo "Started: $(date)" | tee -a $LOG

# ── Phase 3 补跑: Cross-vehicle ──
echo "" | tee -a $LOG
echo "=== Cross-vehicle (kia->sonata) ===" | tee -a $LOG
$PY src/experiments/run_cross_vehicle.py --root . --train-vehicle kia --test-vehicle sonata 2>&1 | tee -a $LOG

# ── CrySyS main experiments ──
echo "" | tee -a $LOG
echo "=== CrySyS main ===" | tee -a $LOG

echo "[CrySyS] Transformer" | tee -a $LOG
$PY src/training/train_baseline.py --root . --dataset crysys --model transformer --epochs 15 2>&1 | tee -a $LOG

echo "[CrySyS] CNN" | tee -a $LOG
$PY src/training/train_baseline.py --root . --dataset crysys --model cnn --epochs 15 2>&1 | tee -a $LOG

echo "[CrySyS] LSTM" | tee -a $LOG
$PY src/training/train_baseline.py --root . --dataset crysys --model lstm --epochs 15 2>&1 | tee -a $LOG

echo "[CrySyS] SCS-CAN w/o SSL" | tee -a $LOG
$PY src/training/finetune.py --root . --dataset crysys --model scs_can --variant wo_ssl --epochs 15 --ipc-weight 0 --table main_results.csv 2>&1 | tee -a $LOG

echo "[CrySyS] Pretrain" | tee -a $LOG
$PY src/training/pretrain.py --root . --dataset crysys --epochs 10 2>&1 | tee -a $LOG

echo "[CrySyS] SCS-CAN full" | tee -a $LOG
$PY src/training/finetune.py --root . --dataset crysys --model scs_can --variant full --epochs 15 --pretrained checkpoints/crysys/pretrain/best.pt --ipc-weight 0.1 --table main_results.csv 2>&1 | tee -a $LOG

echo "" | tee -a $LOG
echo "=== ALL REMAINING DONE ===" | tee -a $LOG
echo "Finished: $(date)" | tee -a $LOG
