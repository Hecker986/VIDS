"""HCRL-CH CSV -> frames.parquet."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT.parent / "vids_proj"))
sys.path.insert(0, str(ROOT))

from src.data_parsers.common import parse_sources_manifest
from src.utils.io import save_frames
from vids_data.loaders import load_cantt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "data/raw/hcrl_ch_sources.txt"))
    ap.add_argument("--out", default=str(ROOT / "data/processed/hcrl_ch/frames.parquet"))
    args = ap.parse_args()
    df = parse_sources_manifest(Path(args.manifest), lambda p: load_cantt(p, "hcrl_ch"), "hcrl_ch")
    save_frames(df, args.out)
    print(f"saved {args.out} rows={len(df)}", flush=True)


if __name__ == "__main__":
    main()
