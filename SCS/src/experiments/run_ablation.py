"""Ablation experiments."""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--dataset", required=True)
    args = ap.parse_args()
    py = sys.executable
    for variant in ("full", "wo_ssl", "wo_mfm", "wo_ipc", "wo_transition"):
        cmd = [py, str(ROOT / "src/training/finetune.py"), "--root", args.root,
               "--dataset", args.dataset, "--variant", variant, "--epochs", "15",
               "--table", "ablation_results.csv", "--ipc-weight", "0"]
        if variant == "full":
            pt = Path(args.root) / "checkpoints" / args.dataset / "pretrain/best.pt"
            if pt.exists():
                cmd += ["--pretrained", str(pt), "--ipc-weight", "0.1"]
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
