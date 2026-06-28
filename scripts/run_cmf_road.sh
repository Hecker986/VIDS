#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
PY="${PY:-$ROOT/.venv/bin/python}"
LOG_DIR="$ROOT/logs/cmf"
mkdir -p "$LOG_DIR" "$ROOT/results/cmf_tables"
LOG="$LOG_DIR/road.log"

echo "=== CMF-CAN ROAD experiments ===" | tee "$LOG"
echo "Started: $(date)" | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== Build/reuse CMF features ===" | tee -a "$LOG"
"$PY" -m cmf_can.data.build_features --root "$ROOT" --dataset road 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== ROAD main experiment ===" | tee -a "$LOG"
for MODEL in cnn lstm gru transformer frame_only stats_only concat_fusion cmf_can; do
  "$PY" -m cmf_can.training.cli \
    --root "$ROOT" \
    --dataset road \
    --model "$MODEL" \
    --epochs "${CMF_MAIN_EPOCHS:-20}" \
    --batch-size "${CMF_BATCH_SIZE:-512}" \
    --table road_main.csv \
    --num-workers "${CMF_NUM_WORKERS:-2}" 2>&1 | tee -a "$LOG"
done

echo "" | tee -a "$LOG"
echo "=== ROAD few-label experiment ===" | tee -a "$LOG"
for SEED in 42 2024 2026; do
  for RATIO in 0.01 0.05 0.10 0.20 1.0; do
    for MODEL in transformer concat_fusion cmf_can; do
      "$PY" -m cmf_can.training.cli \
        --root "$ROOT" \
        --dataset road \
        --model "$MODEL" \
        --epochs "${CMF_FEW_EPOCHS:-15}" \
        --batch-size "${CMF_BATCH_SIZE:-512}" \
        --seed "$SEED" \
        --label-ratio "$RATIO" \
        --table road_few_label.csv \
        --num-workers "${CMF_NUM_WORKERS:-2}" 2>&1 | tee -a "$LOG"
    done
  done
done

echo "" | tee -a "$LOG"
echo "=== ROAD ablation experiment ===" | tee -a "$LOG"
for MODEL in cmf_can wo_stats wo_context wo_xattn wo_gate frame_only stats_only concat_fusion; do
  "$PY" -m cmf_can.training.cli \
    --root "$ROOT" \
    --dataset road \
    --model "$MODEL" \
    --epochs "${CMF_ABLATION_EPOCHS:-15}" \
    --batch-size "${CMF_BATCH_SIZE:-512}" \
    --table road_ablation.csv \
    --num-workers "${CMF_NUM_WORKERS:-2}" 2>&1 | tee -a "$LOG"
done

echo "" | tee -a "$LOG"
echo "Finished: $(date)" | tee -a "$LOG"

