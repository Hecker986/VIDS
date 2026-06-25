"""Record CAN ID vocabulary size from frames."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    df = pd.read_parquet(args.frames)
    vocab = {"num_ids": int(df["can_id"].max()) + 1, "unique_ids": int(df["can_id"].nunique())}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(vocab, indent=2))
    print(vocab, flush=True)


if __name__ == "__main__":
    main()
