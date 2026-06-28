#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
PY="${PY:-$ROOT/.venv/bin/python}"
LOG_DIR="$ROOT/logs/cmf"
mkdir -p "$LOG_DIR" "$ROOT/results/cmf_tables"
LOG="$LOG_DIR/ctt.log"

echo "=== CMF-CAN CT&T experiments ===" | tee "$LOG"
echo "Started: $(date)" | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== Download CT&T set_01 ===" | tee -a "$LOG"
"$PY" scripts/download_cantt.py 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== Prepare CT&T official generalization splits ===" | tee -a "$LOG"
"$PY" -m cmf_can.data.prepare_ctt --root "$ROOT" 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== Build CT&T CMF features ===" | tee -a "$LOG"
for DS in ctt_test01 ctt_test02 ctt_test03 ctt_test04; do
  "$PY" -m cmf_can.data.build_features --root "$ROOT" --dataset "$DS" 2>&1 | tee -a "$LOG"
done

echo "" | tee -a "$LOG"
echo "=== CT&T generalization experiments ===" | tee -a "$LOG"
for DS in ctt_test01 ctt_test02 ctt_test03 ctt_test04; do
  for MODEL in transformer concat_fusion cmf_can; do
    "$PY" -m cmf_can.training.cli \
      --root "$ROOT" \
      --dataset "$DS" \
      --model "$MODEL" \
      --epochs "${CMF_CTT_EPOCHS:-15}" \
      --batch-size "${CMF_BATCH_SIZE:-512}" \
      --table ctt_generalization.csv \
      --num-workers "${CMF_NUM_WORKERS:-2}" 2>&1 | tee -a "$LOG"
  done
done

echo "" | tee -a "$LOG"
echo "Finished: $(date)" | tee -a "$LOG"

