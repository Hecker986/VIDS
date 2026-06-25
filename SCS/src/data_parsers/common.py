"""Convert raw loader output to unified SCS-CAN frame table."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.schema import ATTACK_TO_ID, FRAME_COLUMNS


def _infer_attack(name: str, attack: str) -> str:
    n = name.lower()
    if "masquerade" in n:
        return "masquerade"
    if "gear" in n:
        return "gear_spoof"
    if "rpm" in n:
        return "rpm_spoof"
    if "malfunction" in n or "malfunc" in n:
        return "malfunction"
    if "fuzz" in n:
        return "fuzzing"
    if "dos" in n:
        return "dos"
    if attack in ATTACK_TO_ID:
        return attack
    if attack == "fuzzy":
        return "fuzzing"
    if attack == "fabrication":
        return "spoof"
    return "unknown" if attack not in ("normal",) else "normal"


def _vehicle_from_path(path: str) -> str:
    n = path.lower()
    for v in ("kia", "sonata", "spark"):
        if v in n:
            return v
    return "unknown"


def payload_to_bytes(payload) -> bytes:
    if isinstance(payload, (bytes, bytearray)):
        return bytes(payload)
    if isinstance(payload, str):
        return bytes.fromhex(payload.replace(" ", ""))
    return bytes(payload)


def finalize_frames(df: pd.DataFrame, dataset: str, capture_id: str,
                    vehicle: str = "unknown") -> pd.DataFrame:
    out = pd.DataFrame()
    out["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").astype(np.float64)
    out["can_id"] = pd.to_numeric(df["can_id"], errors="coerce").fillna(0).astype(np.int32) % 4096
    payloads = df["payload"].map(payload_to_bytes)
    out["dlc"] = payloads.map(len).clip(0, 8).astype(np.int8)
    for i in range(8):
        out[f"data{i}"] = payloads.map(lambda b, j=i: b[j] if j < len(b) else 0).astype(np.int16)
    out = out.sort_values("timestamp").reset_index(drop=True)
    out["delta_t_global"] = out["timestamp"].diff().fillna(0.0).astype(np.float32)
    last_ts = {}
    same = []
    for ts, cid in zip(out["timestamp"], out["can_id"]):
        prev = last_ts.get(int(cid))
        same.append(float(ts - prev) if prev is not None else 0.0)
        last_ts[int(cid)] = float(ts)
    out["delta_t_same_id"] = np.asarray(same, dtype=np.float32)
    out["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(np.int8)
    raw_attack = df["attack"] if "attack" in df.columns else "normal"
    out["attack_type"] = [_infer_attack(f"{capture_id} {a}", str(a)) for a in raw_attack]
    out.loc[out["label"] == 0, "attack_type"] = "normal"
    out["dataset"] = dataset
    out["vehicle"] = vehicle
    out["capture_id"] = capture_id
    out["split_group"] = vehicle if vehicle != "unknown" else capture_id
    return out[FRAME_COLUMNS]


def parse_sources_manifest(manifest: Path, loader, dataset: str) -> pd.DataFrame:
    paths = [ln.strip() for ln in manifest.read_text().splitlines() if ln.strip()]
    parts = []
    for p in paths:
        raw = loader(p)
        cap = f"{Path(p).parent.name}/{Path(p).stem}"
        veh = _vehicle_from_path(p)
        parts.append(finalize_frames(raw, dataset, cap, veh))
        print(f"parsed {Path(p).name} {len(raw)}", flush=True)
    return pd.concat(parts, ignore_index=True)
