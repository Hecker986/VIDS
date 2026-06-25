#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
bash scripts/prepare_dataset.sh hcrl_ch 2>&1 | tee logs/day1_prepare_hcrl_ch.log
python src/training/train_baseline.py --dataset hcrl_ch --model transformer --epochs 20 2>&1 | tee logs/day1_transformer_hcrl_ch.log
