#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
DS="${1:-hcrl_ch}"
case "$DS" in
  road)
    MANIFEST="$ROOT/data/raw/road_sources.txt"
    OUT="$ROOT/data/processed/road"
    python src/data_parsers/parse_road_log.py --manifest "$MANIFEST" --out "$OUT/frames.parquet"
    ;;
  hcrl_ch)
    MANIFEST="$ROOT/data/raw/hcrl_ch_sources.txt"
    OUT="$ROOT/data/processed/hcrl_ch"
    python src/data_parsers/parse_hcrl_ch_csv.py --manifest "$MANIFEST" --out "$OUT/frames.parquet"
    ;;
  hcrl_sa)
    MANIFEST="$ROOT/data/raw/hcrl_sa_sources.txt"
    OUT="$ROOT/data/processed/hcrl_sa"
    python src/data_parsers/parse_hcrl_sa_csv.py --manifest "$MANIFEST" --out "$OUT/frames.parquet"
    ;;
  *) echo "unknown dataset $DS"; exit 1 ;;
esac
python src/preprocessing/build_vocab.py --frames "$OUT/frames.parquet" --out "$OUT/vocab.json"
python src/preprocessing/build_splits.py --frames "$OUT/frames.parquet" --out "$OUT/splits.json"
python src/preprocessing/build_window_index.py --frames "$OUT/frames.parquet" --splits "$OUT/splits.json" --out "$OUT/windows_index.npy"
python src/preprocessing/build_transition_graph.py --frames "$OUT/frames.parquet" --windows "$OUT/windows_index.npy" --out-dir "$OUT"
python src/preprocessing/compute_train_stats.py --frames "$OUT/frames.parquet" --windows "$OUT/windows_index.npy" --out "$OUT/train_stats.json"
echo "prepared $DS"
