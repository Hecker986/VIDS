# Diagnostic Command Log

This file records the commands used for the first-principles CMF-CAN diagnostic package.

## Directory and Inventory

```bash
mkdir -p results/cmf_diagnostics
mkdir -p results/cmf_predictions
mkdir -p results/cmf_diagnostics/figures
mkdir -p results/cmf_diagnostics/tables
mkdir -p results/cmf_diagnostics/logs
find results -maxdepth 3 -type f | sort > results/cmf_diagnostics/current_file_inventory.txt
```

## Prediction Metadata Repair

Existing prediction dumps were real per-sample outputs, but they did not include the `threshold` column requested for D1. Thresholds were backfilled from the matching aggregate result tables without changing labels, scores, or predictions.

Sources:

- `results/cmf_tables/road_main_20ep.csv`
- `results/cmf_tables/ctt_generalization_15ep.csv`

Log:

- `results/cmf_diagnostics/logs/threshold_backfill.log`

## Diagnostic Generation

```bash
.venv/bin/python -m cmf_can.analysis.first_principles_diagnostics
```

## Consistency Audit

```bash
.venv/bin/python - <<'PY'
# Computed results/cmf_diagnostics/tables/prediction_table_consistency_audit.csv
# and results/cmf_diagnostics/prediction_table_consistency_audit.md.
PY
```

## Verification

```bash
.venv/bin/python -m compileall cmf_can/analysis cmf_can/training
find results/cmf_diagnostics -type f | sort > results/cmf_diagnostics/diagnostics_inventory.txt
git status --short
```
