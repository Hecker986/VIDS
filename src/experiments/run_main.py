"""Run main performance experiments."""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def run(cmd: list[str]) -> None:
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    root = Path(args.root)
    py = sys.executable
    epochs = "10" if args.quick else "20"
    for model in ("cnn", "lstm", "transformer"):
        run([py, str(root / "src/training/train_baseline.py"), "--root", str(root),
             "--dataset", args.dataset, "--model", model, "--epochs", epochs])
    run([py, str(root / "src/training/finetune.py"), "--root", str(root),
         "--dataset", args.dataset, "--variant", "wo_ssl", "--epochs", epochs, "--ipc-weight", "0"])
    if not args.quick:
        run([py, str(root / "src/training/pretrain.py"), "--root", str(root),
             "--dataset", args.dataset, "--epochs", "10"])
        pt = root / "checkpoints" / args.dataset / "pretrain/best.pt"
        run([py, str(root / "src/training/finetune.py"), "--root", str(root),
             "--dataset", args.dataset, "--variant", "full", "--pretrained", str(pt), "--epochs", epochs])


if __name__ == "__main__":
    main()
