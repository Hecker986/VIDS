#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

python src/experiments/run_few_label.py --dataset hcrl_ch 2>&1 | tee logs/day4_few_label_hcrl_ch_retry.log
python src/experiments/run_few_label.py --dataset road 2>&1 | tee logs/day4_few_label_road_retry.log
python src/experiments/run_cross_vehicle.py --train-vehicle kia --test-vehicle sonata 2>&1 | tee logs/day5_cross_vehicle_retry.log

echo "Day4-5 retry complete"
