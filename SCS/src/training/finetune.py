"""SCS-CAN finetune."""
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
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", default="scs_can")
    ap.add_argument("--variant", default="full", choices=("full", "wo_ssl", "wo_mfm", "wo_ipc", "wo_transition"))
    ap.add_argument("--pretrained", default="")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--label-ratio", type=float, default=1.0)
    ap.add_argument("--ipc-weight", type=float, default=0.1)
    ap.add_argument("--table", default="main_results.csv")
    args = ap.parse_args()
    root = Path(args.root)
    pt = Path(args.pretrained) if args.pretrained else None
    ipc = args.ipc_weight if args.variant == "full" else 0.0
    result = finetune(root, args.dataset, args.model, variant=args.variant,
                      epochs=args.epochs, batch_size=args.batch_size, seed=args.seed,
                      label_ratio=args.label_ratio, pretrained=pt, ipc_weight=ipc)
    out = root / "results/tables" / args.table
    out.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out.exists()
    with out.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(result.keys()))
        if write_header:
            w.writeheader()
        w.writerow(result)


if __name__ == "__main__":
    main()
