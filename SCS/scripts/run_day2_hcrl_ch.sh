#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
python src/training/finetune.py --dataset hcrl_ch --variant wo_ssl --epochs 20 --ipc-weight 0 2>&1 | tee logs/day2_scs_can_wo_ssl_hcrl_ch.log
