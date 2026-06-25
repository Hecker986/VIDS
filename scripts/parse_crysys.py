"""Parse raw CrySyS candump logs → SCS-CAN frames.parquet (vectorized)."""
from __future__ import annotations
import json, re, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/root/autodl-tmp/datasets_extra/crysys_ds/extracted/"
            "CrySyS dataset of CAN traffic logs containing fabrication "
            "and masquerade attacks")
OUT = Path("/root/autodl-tmp/scs-can/data/processed/crysys")

LINE_RE = re.compile(rb"^\((\d+\.\d+)\)\s+\S+\s+([0-9A-Fa-f]+)#([0-9A-Fa-fR]*)")


def parse_log(path: Path):
    ids, payloads, timestamps = [], [], []
    with open(path, "rb") as f:
        for raw in f:
            m = LINE_RE.match(raw)
            if not m: continue
            timestamps.append(float(m.group(1)))
            ids.append(int(m.group(2), 16))
            data = m.group(3)
            if data.startswith(b"R") or len(data) == 0:
                payloads.append(b"\x00" * 8)
            else:
                h = data.decode("ascii", errors="replace")
                if len(h) % 2: h += "0"
                h = h[:16].ljust(16, "0")
                payloads.append(bytes.fromhex(h))
    if not ids:
        return np.empty(0), np.empty(0, dtype=np.int32), np.empty((0,8), dtype=np.uint8)
    ts = np.array(timestamps, dtype=np.float64)
    cid = np.array(ids, dtype=np.int32)
    pay = np.frombuffer(b"".join(payloads), dtype=np.uint8).reshape(-1, 8)
    return ts, cid, pay


def extract_attack_window(json_path: Path):
    try:
        meta = json.loads(json_path.read_text())
    except: return None
    markers = meta.get("markers", [])
    if len(markers) < 2: return None
    start_t, end_t, target_id = None, None, None
    for mk in markers:
        desc = (mk.get("description") or "").lower()
        t = mk.get("time")
        pid = mk.get("packet_ID")
        if t is None: continue
        if target_id is None and pid:
            try: target_id = int(str(pid), 16)
            except: pass
        if "start" in desc: start_t = float(t)
        elif "end" in desc: end_t = float(t)
    if start_t is None or end_t is None or target_id is None: return None
    return start_t, end_t, target_id


def attack_class(stem: str) -> str:
    s = stem.lower()
    if "msg-inj" in s or "msg_inj" in s: return "fabrication"
    if "msg-mod" in s or "msg_mod" in s: return "masquerade"
    return "fabrication"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rec_dirs = sorted([d for d in ROOT.iterdir() if d.is_dir() and
                       (d.name.startswith("S-") or d.name.startswith("T-"))])
    print(f"Found {len(rec_dirs)} recording dirs", flush=True)

    dfs = []
    total_frames = 0
    for di, d in enumerate(rec_dirs):
        logs = sorted(d.glob("*.log"))
        for log_path in logs:
            if "inj-messages" in log_path.name: continue
            is_benign = "benign" in log_path.name.lower()
            ts, ids, pay = parse_log(log_path)
            n = len(ts)
            if n == 0: continue

            labels = np.zeros(n, dtype=np.int32)
            at_arr = np.full(n, "normal", dtype=object)

            if not is_benign:
                jp = log_path.with_suffix(".json")
                aw = extract_attack_window(jp) if jp.exists() else None
                at = attack_class(log_path.stem)
                if aw:
                    s_t, e_t, tid = aw
                    mask = (ts >= s_t) & (ts <= e_t) & (ids == tid)
                    labels[mask] = 1
                    at_arr[mask] = at

            dt_global = np.zeros(n, dtype=np.float32)
            dt_global[1:] = (ts[1:] - ts[:-1]).astype(np.float32)
            dt_same_id = np.zeros(n, dtype=np.float32)
            last_ts = {}
            for i in range(n):
                cid = int(ids[i])
                if cid in last_ts:
                    dt_same_id[i] = ts[i] - last_ts[cid]
                last_ts[cid] = ts[i]

            df = pd.DataFrame({
                "timestamp": ts,
                "can_id": ids.astype(np.int32),
                "dlc": np.full(n, 8, dtype=np.int8),
                "data0": pay[:, 0], "data1": pay[:, 1],
                "data2": pay[:, 2], "data3": pay[:, 3],
                "data4": pay[:, 4], "data5": pay[:, 5],
                "data6": pay[:, 6], "data7": pay[:, 7],
                "delta_t_global": dt_global,
                "delta_t_same_id": dt_same_id,
                "label": labels,
                "attack_type": at_arr,
                "dataset": "crysys",
                "vehicle": "crysys",
                "capture_id": log_path.stem,
                "split_group": d.name,
            })
            dfs.append(df)
            total_frames += n
            atk = int(labels.sum())
            print(f"  [{di+1}/{len(rec_dirs)}] {log_path.name}: {n} frames, atk={atk}", flush=True)

    print(f"Concatenating {total_frames} frames...", flush=True)
    full = pd.concat(dfs, ignore_index=True)
    print(f"Label dist: {full['label'].value_counts().to_dict()}", flush=True)
    print(f"Attack types: {full['attack_type'].value_counts().to_dict()}", flush=True)
    full.to_parquet(OUT / "frames.parquet", index=False)
    print(f"Saved {OUT / 'frames.parquet'} ({len(full)} rows)", flush=True)


if __name__ == "__main__":
    main()
