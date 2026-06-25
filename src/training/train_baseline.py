"""Train baseline models."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.training.trainer import finetune


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--dataset", required=True, choices=("road", "hcrl_ch", "hcrl_sa", "crysys", "crysys_subset"))
    ap.add_argument("--model", required=True, choices=("cnn", "lstm", "transformer"))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    root = Path(args.root)
    result = finetune(root, args.dataset, args.model, epochs=args.epochs,
                      batch_size=args.batch_size, seed=args.seed)
    out = root / "results/tables/main_results.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out.exists()
    with out.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(result.keys()))
        if write_header:
            w.writeheader()
        w.writerow(result)


if __name__ == "__main__":
    main()
