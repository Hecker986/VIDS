"""Unified CAN frame schema and attack-type vocabulary."""
from __future__ import annotations

import numpy as np

ATTACK_TYPES = [
    "normal", "masquerade", "spoof", "gear_spoof", "rpm_spoof",
    "malfunction", "fuzzing", "dos", "unknown", "fabrication",
]
ATTACK_TO_ID = {a: i for i, a in enumerate(ATTACK_TYPES)}
ATTACK_PRIORITY = {a: i for i, a in enumerate(reversed(ATTACK_TYPES))}

FRAME_COLUMNS = [
    "timestamp", "can_id", "dlc",
    "data0", "data1", "data2", "data3", "data4", "data5", "data6", "data7",
    "delta_t_global", "delta_t_same_id",
    "label", "attack_type", "dataset", "vehicle", "capture_id", "split_group",
]

SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST = 0, 1, 2
WINDOW_SIZE, STRIDE, PAYLOAD_LEN = 100, 100, 8
VOCAB_SIZE = 4096
TOP_K_TRANSITION = 16


def contiguous_runs(pos: np.ndarray) -> list[np.ndarray]:
    if len(pos) == 0:
        return []
    runs, start = [], 0
    for i in range(1, len(pos)):
        if pos[i] != pos[i - 1] + 1:
            runs.append(pos[start:i])
            start = i
    runs.append(pos[start:])
    return runs
