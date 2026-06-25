#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
python src/training/pretrain.py --dataset hcrl_ch --epochs 10 2>&1 | tee logs/day3_pretrain_hcrl_ch.log
python src/training/finetune.py --dataset hcrl_ch --variant full --pretrained checkpoints/hcrl_ch/pretrain/best.pt --epochs 20 2>&1 | tee logs/day3_full_scs_can_hcrl_ch.log
