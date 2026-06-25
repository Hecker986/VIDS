#!/usr/bin/env bash
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
PY=python
LAST_FAIL=0

run_step() {
  local name="$1"
  shift
  echo "=== START: $name ==="
  if "$@"; then
    echo "=== OK: $name ==="
  else
    LAST_FAIL=$?
    echo "=== FAILED: $name (exit $LAST_FAIL) ===" >&2
  fi
}

run_step day4_few_label_hcrl_ch bash -c "$PY src/experiments/run_few_label.py --dataset hcrl_ch 2>&1 | tee logs/day4_few_label_hcrl_ch.log"
run_step day4_few_label_road bash -c "$PY src/experiments/run_few_label.py --dataset road 2>&1 | tee logs/day4_few_label_road.log"
run_step day5_cross_vehicle bash -c "$PY src/experiments/run_cross_vehicle.py --train-vehicle kia --test-vehicle sonata 2>&1 | tee logs/day5_cross_vehicle.log"

for ds in hcrl_ch road; do
  for m in cnn lstm; do
    run_step "day6_${m}_${ds}" bash -c "$PY src/training/train_baseline.py --dataset \"$ds\" --model \"$m\" --epochs 15 --batch-size 512 2>&1 | tee \"logs/day6_${m}_${ds}.log\""
  done
  run_step "day6_ablation_${ds}" bash -c "$PY src/experiments/run_ablation.py --dataset \"$ds\" 2>&1 | tee \"logs/day6_ablation_${ds}.log\""
done

if [[ "$LAST_FAIL" -ne 0 ]]; then
  echo "Day4-6 finished with failures (last exit $LAST_FAIL)"
  exit "$LAST_FAIL"
fi
echo "Day4-6 complete"
