#!/bin/bash
set -e
cd /root/autodl-tmp/scs-can
PY=/root/miniconda3/bin/python
LOG=logs/run_all_fixed.log
mkdir -p logs results/tables

echo "=== SCS-CAN Full Experiment Run ===" | tee $LOG
echo "Started: $(date)" | tee -a $LOG

# ── Phase 1: ROAD main (5 models) ──
echo "" | tee -a $LOG
echo "=== Phase 1: ROAD main ===" | tee -a $LOG

echo "[ROAD] Transformer" | tee -a $LOG
$PY src/training/train_baseline.py --root . --dataset road --model transformer --epochs 30 2>&1 | tee -a $LOG

echo "[ROAD] CNN" | tee -a $LOG
$PY src/training/train_baseline.py --root . --dataset road --model cnn --epochs 30 2>&1 | tee -a $LOG

echo "[ROAD] LSTM" | tee -a $LOG
$PY src/training/train_baseline.py --root . --dataset road --model lstm --epochs 30 2>&1 | tee -a $LOG

echo "[ROAD] SCS-CAN w/o SSL" | tee -a $LOG
$PY src/training/finetune.py --root . --dataset road --model scs_can --variant wo_ssl --epochs 30 --ipc-weight 0 --table main_results.csv 2>&1 | tee -a $LOG

echo "[ROAD] Pretrain" | tee -a $LOG
$PY src/training/pretrain.py --root . --dataset road --epochs 20 2>&1 | tee -a $LOG

echo "[ROAD] SCS-CAN full" | tee -a $LOG
$PY src/training/finetune.py --root . --dataset road --model scs_can --variant full --epochs 30 --pretrained checkpoints/road/pretrain/best.pt --ipc-weight 0.1 --table main_results.csv 2>&1 | tee -a $LOG

# ── Phase 2: HCRL-CH main ──
echo "" | tee -a $LOG
echo "=== Phase 2: HCRL-CH main ===" | tee -a $LOG

echo "[HCRL-CH] Transformer" | tee -a $LOG
$PY src/training/train_baseline.py --root . --dataset hcrl_ch --model transformer --epochs 30 2>&1 | tee -a $LOG

echo "[HCRL-CH] CNN" | tee -a $LOG
$PY src/training/train_baseline.py --root . --dataset hcrl_ch --model cnn --epochs 30 2>&1 | tee -a $LOG

echo "[HCRL-CH] LSTM" | tee -a $LOG
$PY src/training/train_baseline.py --root . --dataset hcrl_ch --model lstm --epochs 30 2>&1 | tee -a $LOG

echo "[HCRL-CH] SCS-CAN w/o SSL" | tee -a $LOG
$PY src/training/finetune.py --root . --dataset hcrl_ch --model scs_can --variant wo_ssl --epochs 30 --ipc-weight 0 --table main_results.csv 2>&1 | tee -a $LOG

echo "[HCRL-CH] Pretrain" | tee -a $LOG
$PY src/training/pretrain.py --root . --dataset hcrl_ch --epochs 20 2>&1 | tee -a $LOG

echo "[HCRL-CH] SCS-CAN full" | tee -a $LOG
$PY src/training/finetune.py --root . --dataset hcrl_ch --model scs_can --variant full --epochs 30 --pretrained checkpoints/hcrl_ch/pretrain/best.pt --ipc-weight 0.1 --table main_results.csv 2>&1 | tee -a $LOG

# ── Phase 3: Cross-vehicle ──
echo "" | tee -a $LOG
echo "=== Phase 3: Cross-vehicle ===" | tee -a $LOG
$PY src/experiments/run_cross_vehicle.py --root . --train-vehicle kia --test-vehicle sonata 2>&1 | tee -a $LOG

# ── Phase 4: Few-label (ROAD) ──
echo "" | tee -a $LOG
echo "=== Phase 4: Few-label ROAD ===" | tee -a $LOG
$PY src/experiments/run_few_label.py --root . --dataset road 2>&1 | tee -a $LOG

# ── Phase 5: Ablation (ROAD) ──
echo "" | tee -a $LOG
echo "=== Phase 5: Ablation ROAD ===" | tee -a $LOG
$PY src/experiments/run_ablation.py --root . --dataset road 2>&1 | tee -a $LOG

echo "" | tee -a $LOG
echo "=== ALL DONE ===" | tee -a $LOG
echo "Finished: $(date)" | tee -a $LOG
