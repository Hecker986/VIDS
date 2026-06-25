"""Frame table I/O."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_frames(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".npz":
        z = np.load(path, allow_pickle=False)
        return pd.DataFrame({k: z[k] for k in z.files})
    raise ValueError(f"unsupported frames format: {path}")


def save_frames(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
