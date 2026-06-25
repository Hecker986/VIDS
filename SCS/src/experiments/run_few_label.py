"""Few-label experiments.

Only SCS-CAN full loads pretrained weights. Transformer and SCS-CAN w/o SSL
train from scratch at each label ratio.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--dataset", required=True)
    args = ap.parse_args()
    py = sys.executable
    for ratio in (0.01, 0.05, 0.10, 0.20, 1.0):
        for model, variant in [("transformer", "full"),
                               ("scs_can", "wo_ssl"),
                               ("scs_can", "full")]:
            cmd = [py, str(ROOT / "src/training/finetune.py"),
                   "--root", args.root,
                   "--dataset", args.dataset,
                   "--model", model, "--variant", variant,
                   "--label-ratio", str(ratio),
                   "--epochs", "15",
                   "--table", "few_label_results.csv"]
            # Only load pretrained for scs_can full
            if model == "scs_can" and variant == "full":
                pt = Path(args.root) / "checkpoints" / args.dataset / "pretrain/best.pt"
                if pt.exists():
                    cmd += ["--pretrained", str(pt)]
            print(" ".join(cmd), flush=True)
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
