"""SCS-CAN pretrain."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.training.trainer import pretrain


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    path = pretrain(Path(args.root), args.dataset, epochs=args.epochs,
                    batch_size=args.batch_size, seed=args.seed)
    print(f"saved {path}", flush=True)


if __name__ == "__main__":
    main()
