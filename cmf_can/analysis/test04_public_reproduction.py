from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.cluster import Birch
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    IsolationForest,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


ROOT = Path(".")
RAW_ROOT = ROOT / "data/raw/can-train-and-test/set_01"
OUT = ROOT / "results/test04_public_reproduction"
TABLES = OUT / "tables"
FIGS = OUT / "figures"
LOGS = OUT / "logs"
PREDS = OUT / "predictions"
AUDITS = OUT / "audits"
CONFIGS = OUT / "configs"
MANIFESTS = OUT / "manifests"

TEST_FOLDERS = {
    "ctt_test01": "test_01_known_vehicle_known_attack",
    "ctt_test02": "test_02_unknown_vehicle_known_attack",
    "ctt_test03": "test_03_known_vehicle_unknown_attack",
    "ctt_test04": "test_04_unknown_vehicle_unknown_attack",
}

FPR_BUDGETS = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]


@dataclass(frozen=True)
class FeatureProtocol:
    name: str
    columns: tuple[str, ...]
    safe: bool
    notes: str


PROTOCOLS: dict[str, FeatureProtocol] = {
    "P0_current": FeatureProtocol(
        "P0_current",
        (
            "can_id",
            "dlc",
            "data0",
            "data1",
            "data2",
            "data3",
            "data4",
            "data5",
            "data6",
            "data7",
            "delta_t_global",
            "delta_t_same_id",
            "payload_sum",
            "payload_mean",
            "payload_std",
            "payload_delta_l1",
        ),
        True,
        "Current final_grain_can sample-level feature set.",
    ),
    "P1_public_default": FeatureProtocol(
        "P1_public_default",
        ("timestamp", "can_id", "data0", "data1", "data2", "data3", "data4", "data5", "data6", "data7"),
        False,
        "Public-style timestamp + arbitration ID + split payload bytes.",
    ),
    "P2_no_timestamp": FeatureProtocol(
        "P2_no_timestamp",
        ("can_id", "data0", "data1", "data2", "data3", "data4", "data5", "data6", "data7"),
        True,
        "Public payload/ID protocol with raw timestamp removed.",
    ),
    "P3_no_subdivision": FeatureProtocol(
        "P3_no_subdivision",
        ("timestamp", "can_id", "data_int"),
        False,
        "Raw timestamp + arbitration ID + 8-byte payload interpreted as a single integer.",
    ),
    "P4_timestamp_only": FeatureProtocol(
        "P4_timestamp_only",
        ("timestamp",),
        False,
        "Raw timestamp only, shortcut audit.",
    ),
    "P5_arbitration_payload_only": FeatureProtocol(
        "P5_arbitration_payload_only",
        ("can_id", "data_int"),
        True,
        "Arbitration ID + payload as one numeric field, no raw timestamp.",
    ),
    "P6_delta_features": FeatureProtocol(
        "P6_delta_features",
        ("delta_t_global", "delta_t_same_id", "payload_delta_l1", "payload_sum", "payload_mean", "payload_std"),
        True,
        "Causal timing/payload-delta statistics only.",
    ),
    "P7_public_plus_delta": FeatureProtocol(
        "P7_public_plus_delta",
        (
            "timestamp",
            "can_id",
            "data0",
            "data1",
            "data2",
            "data3",
            "data4",
            "data5",
            "data6",
            "data7",
            "delta_t_global",
            "delta_t_same_id",
            "payload_delta_l1",
            "payload_sum",
            "payload_mean",
            "payload_std",
        ),
        False,
        "Public default plus causal delta/payload statistics.",
    ),
    "SAFE_CAN": FeatureProtocol(
        "SAFE_CAN",
        (
            "can_id",
            "dlc",
            "data0",
            "data1",
            "data2",
            "data3",
            "data4",
            "data5",
            "data6",
            "data7",
            "delta_t_global",
            "delta_t_same_id",
            "payload_delta_l1",
            "payload_sum",
            "payload_mean",
            "payload_std",
        ),
        True,
        "Safe CAN-only features; no raw timestamp, sample index, file ID, or labels.",
    ),
    "RISKY_PROTOCOL": FeatureProtocol(
        "RISKY_PROTOCOL",
        ("timestamp", "sample_index", "file_hash"),
        False,
        "Risky protocol/capture-schedule features for shortcut audit only.",
    ),
}


def setup() -> None:
    for p in [TABLES, FIGS, LOGS, PREDS, AUDITS, CONFIGS, MANIFESTS]:
        p.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.linewidth": 0.9,
            "svg.fonttype": "none",
        }
    )


def payload_bytes(series: pd.Series) -> np.ndarray:
    text = series.fillna("").astype(str).str.replace(" ", "", regex=False).str.upper().to_numpy()
    out = np.zeros((len(text), 8), dtype=np.float32)
    for i, value in enumerate(text):
        value = value[:16].ljust(16, "0")
        for j in range(8):
            try:
                out[i, j] = int(value[j * 2 : j * 2 + 2], 16)
            except ValueError:
                out[i, j] = 0
    return out


def payload_int(series: pd.Series) -> np.ndarray:
    text = series.fillna("").astype(str).str.replace(" ", "", regex=False).str.upper().to_numpy()
    out = np.zeros(len(text), dtype=np.float64)
    denom = float(16**16 - 1)
    for i, value in enumerate(text):
        value = value[:16].ljust(16, "0")
        try:
            out[i] = int(value, 16) / denom
        except ValueError:
            out[i] = 0.0
    return out.astype(np.float32)


def parse_can_id(series: pd.Series) -> np.ndarray:
    out = np.zeros(len(series), dtype=np.float32)
    for i, value in enumerate(series.astype(str)):
        value = value.strip()
        try:
            out[i] = int(value, 16)
        except ValueError:
            out[i] = float(value)
    return out


def stable_file_hash(path: Path) -> float:
    digest = hashlib.md5(str(path).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / float(16**8 - 1)


def read_raw_features(path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(path)
    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    ts = df["timestamp"].astype(float).to_numpy(np.float64)
    payload = payload_bytes(df["data_field"])
    can_id = parse_can_id(df["arbitration_id"])
    dlc = df["data_field"].fillna("").astype(str).str.replace(" ", "", regex=False).str.len().clip(0, 16).to_numpy(np.float32) / 2.0
    delta_global = np.zeros(len(df), dtype=np.float32)
    if len(df) > 1:
        delta_global[1:] = np.maximum(np.diff(ts), 0.0).astype(np.float32)
    delta_same = np.zeros(len(df), dtype=np.float32)
    payload_delta = np.zeros(len(df), dtype=np.float32)
    last_ts: dict[str, float] = {}
    last_payload: dict[str, np.ndarray] = {}
    for i, (cid, t) in enumerate(zip(df["arbitration_id"].astype(str), ts)):
        if cid in last_ts:
            delta_same[i] = max(float(t - last_ts[cid]), 0.0)
            payload_delta[i] = float(np.abs(payload[i] - last_payload[cid]).sum())
        last_ts[cid] = float(t)
        last_payload[cid] = payload[i].copy()
    out = pd.DataFrame(
        {
            "timestamp": ts,
            "relative_time_in_file": ts - ts.min() if len(ts) else ts,
            "sample_index": np.arange(len(df), dtype=np.float32),
            "file_hash": stable_file_hash(path),
            "can_id": can_id,
            "dlc": dlc,
            "delta_t_global": np.log1p(delta_global),
            "delta_t_same_id": np.log1p(delta_same),
            "payload_sum": payload.sum(axis=1),
            "payload_mean": payload.mean(axis=1),
            "payload_std": payload.std(axis=1),
            "payload_delta_l1": payload_delta,
            "data_int": payload_int(df["data_field"]),
        }
    )
    for i in range(8):
        out[f"data{i}"] = payload[:, i]
    y = df["attack"].astype(int).clip(0, 1).to_numpy(np.int8)
    return out, y


def protocol_matrix(frame: pd.DataFrame, protocol: str) -> np.ndarray:
    cols = PROTOCOLS[protocol].columns
    return frame.loc[:, cols].to_numpy(np.float32, copy=True)


def train_val_files() -> tuple[list[Path], list[Path]]:
    files = sorted((RAW_ROOT / "train_01").glob("*.csv"))
    train = [p for p in files if p.stem.endswith("-1")]
    val = [p for p in files if p.stem.endswith("-2")]
    return train, val


def sample_indices(y: np.ndarray, cap_per_file: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    if len(neg) > cap_per_file:
        neg = rng.choice(neg, cap_per_file, replace=False)
    idx = np.concatenate([pos, neg])
    rng.shuffle(idx)
    return idx


def collect_split(protocol: str, neg_cap: int, val_cap: int, seed: int):
    train_files, val_files = train_val_files()
    x_train: list[np.ndarray] = []
    y_train: list[np.ndarray] = []
    x_val: list[np.ndarray] = []
    y_val: list[np.ndarray] = []
    for path in train_files:
        print(f"[public_repro] read train {protocol} {path}", flush=True)
        frame, y = read_raw_features(path)
        idx = sample_indices(y, max(1, neg_cap // max(len(train_files), 1)), seed)
        x_train.append(protocol_matrix(frame, protocol)[idx])
        y_train.append(y[idx])
    for path in val_files:
        print(f"[public_repro] read val {protocol} {path}", flush=True)
        frame, y = read_raw_features(path)
        idx = sample_indices(y, max(1, val_cap // max(len(val_files), 1)), seed)
        x_val.append(protocol_matrix(frame, protocol)[idx])
        y_val.append(y[idx])
    return np.vstack(x_train), np.concatenate(y_train), np.vstack(x_val), np.concatenate(y_val)


def test_paths(setting: str) -> list[Path]:
    return sorted((RAW_ROOT / TEST_FOLDERS[setting]).glob("*.csv"))


def iter_test_features(setting: str, protocol: str):
    for path in test_paths(setting):
        frame, y = read_raw_features(path)
        yield path, protocol_matrix(frame, protocol), y, frame


def metric_values(y: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    pred = score >= threshold
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    spec_prec = tn / max(tn + fn, 1)
    spec_rec = tn / max(tn + fp, 1)
    normal_f1 = 2 * spec_prec * spec_rec / max(spec_prec + spec_rec, 1e-12)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_f1": 0.5 * (f1 + normal_f1),
        "auroc": roc_auc_score(y, score) if len(np.unique(y)) == 2 else np.nan,
        "aupr": average_precision_score(y, score) if len(np.unique(y)) == 2 else np.nan,
        "fpr": fp / max(fp + tn, 1),
        "fnr": fn / max(fn + tp, 1),
        "num_test_pos": int((y == 1).sum()),
        "num_test_neg": int((y == 0).sum()),
    }


def best_f1_threshold(y: np.ndarray, score: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y, score)
    if len(thresholds) == 0:
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.nanargmax(f1))])


def recall_at_budget(y: np.ndarray, score: np.ndarray, budget: float) -> dict:
    order = np.argsort(-score)
    y_sorted = y[order]
    s_sorted = score[order]
    pos = max(int((y == 1).sum()), 1)
    neg = max(int((y == 0).sum()), 1)
    tp = 0
    fp = 0
    best = {"recall": 0.0, "precision": 1.0, "f1": 0.0, "actual_fpr": 0.0, "threshold": np.inf}
    for label, threshold in zip(y_sorted, s_sorted):
        if int(label) == 1:
            tp += 1
        else:
            fp += 1
        fpr = fp / neg
        if fpr <= budget:
            rec = tp / pos
            prec = tp / max(tp + fp, 1)
            best = {
                "recall": rec,
                "precision": prec,
                "f1": 2 * prec * rec / max(prec + rec, 1e-12),
                "actual_fpr": fpr,
                "threshold": float(threshold),
            }
        else:
            break
    return best


def model_specs(seed: int) -> dict[str, BaseEstimator | None]:
    return {
        "MLP": MLPClassifier(hidden_layer_sizes=(64,), max_iter=30, early_stopping=True, random_state=seed),
        "BIRCH": Birch(threshold=2.0, n_clusters=None),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=40, max_depth=2, random_state=seed),
        "RandomForest": RandomForestClassifier(n_estimators=30, min_samples_leaf=2, n_jobs=-1, class_weight="balanced_subsample", random_state=seed),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=30, min_samples_leaf=2, n_jobs=-1, class_weight="balanced", random_state=seed),
        "LogisticRegression": LogisticRegression(max_iter=200, class_weight="balanced"),
        "GaussianNB": GaussianNB(),
        "IsolationForest": IsolationForest(n_estimators=60, contamination="auto", n_jobs=-1, random_state=seed),
        "LinearSVM": SGDClassifier(loss="hinge", class_weight="balanced", max_iter=1000, tol=1e-3, random_state=seed),
        "KNN": None,
        "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=80, max_leaf_nodes=31, random_state=seed),
        "XGBoost": None,
        "LightGBM": None,
        "CatBoost": None,
    }


def score_model(model: BaseEstimator, x: np.ndarray, model_name: str) -> np.ndarray:
    if model_name == "BIRCH":
        centers = getattr(model, "subcluster_centers_", np.empty((0, x.shape[1])))
        if len(centers) == 0:
            return np.zeros(len(x), dtype=np.float32)
        out = np.empty(len(x), dtype=np.float32)
        for start in range(0, len(x), 50_000):
            part = x[start : start + 50_000]
            dist = ((part[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            out[start : start + 50_000] = np.sqrt(dist.min(axis=1))
        return out
    if model_name == "IsolationForest":
        return -model.decision_function(x)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    if hasattr(model, "decision_function"):
        raw = model.decision_function(x)
        return raw.astype(np.float32)
    return model.predict(x).astype(np.float32)


def fit_models(protocol: str, seed: int, neg_cap: int, val_cap: int, models_to_run: Iterable[str]):
    x_train, y_train, x_val, y_val = collect_split(protocol, neg_cap=neg_cap, val_cap=val_cap, seed=seed)
    scaler = StandardScaler().fit(x_train)
    xs = scaler.transform(x_train)
    xv = scaler.transform(x_val)
    normal_xs = xs[y_train == 0]
    fitted: dict[str, BaseEstimator] = {}
    thresholds: dict[str, float] = {}
    fit_times: dict[str, float] = {}
    status_rows = []
    specs = model_specs(seed)
    for name in models_to_run:
        model = specs.get(name)
        if model is None:
            status_rows.append(
                {
                    "feature_protocol": protocol,
                    "model": name,
                    "seed": seed,
                    "status": "not_feasible_or_unavailable",
                    "notes": "KNN/external package model is unavailable or infeasible for full CT&T test scale.",
                }
            )
            continue
        print(f"[public_repro] fit {protocol} {name} seed={seed}", flush=True)
        start = time.time()
        fit_x = normal_xs if name in {"IsolationForest", "BIRCH"} else xs
        try:
            model.fit(fit_x, y_train if name not in {"IsolationForest", "BIRCH"} else None)
            fit_times[name] = time.time() - start
            val_score = score_model(model, xv, name)
            thresholds[name] = best_f1_threshold(y_val, val_score)
            fitted[name] = model
        except Exception as exc:
            status_rows.append(
                {
                    "feature_protocol": protocol,
                    "model": name,
                    "seed": seed,
                    "status": "failed",
                    "notes": f"{type(exc).__name__}: {exc}",
                }
            )
    return scaler, fitted, thresholds, fit_times, x_train, y_train, x_val, y_val, status_rows


def evaluate_models(
    protocol: str,
    scaler: StandardScaler,
    fitted: dict[str, BaseEstimator],
    thresholds: dict[str, float],
    fit_times: dict[str, float],
    y_train: np.ndarray,
    neg_protocol: str,
    seed: int,
    settings: Iterable[str],
) -> list[dict]:
    rows: list[dict] = []
    for setting in settings:
        print(f"[public_repro] eval {protocol} setting={setting}", flush=True)
        scores: dict[str, list[np.ndarray]] = {name: [] for name in fitted}
        y_parts: list[np.ndarray] = []
        start_eval = time.time()
        for _, x_raw, y, _ in iter_test_features(setting, protocol):
            x = scaler.transform(x_raw)
            y_parts.append(y)
            for name, model in fitted.items():
                scores[name].append(score_model(model, x, name))
        y_all = np.concatenate(y_parts)
        infer_time = time.time() - start_eval
        for name, parts in scores.items():
            score = np.concatenate(parts)
            m = metric_values(y_all, score, thresholds[name])
            rows.append(
                {
                    "feature_protocol": protocol,
                    "model": name,
                    "negative_protocol": neg_protocol,
                    "seed": seed,
                    "setting": setting,
                    **m,
                    "num_train_pos": int((y_train == 1).sum()),
                    "num_train_neg": int((y_train == 0).sum()),
                    "fit_time": fit_times[name],
                    "inference_time": infer_time,
                    "threshold": thresholds[name],
                    "status": "completed",
                    "notes": PROTOCOLS[protocol].notes,
                }
            )
            if setting == "ctt_test04" and protocol in {"P1_public_default", "P2_no_timestamp", "P4_timestamp_only", "SAFE_CAN", "RISKY_PROTOCOL", "P7_public_plus_delta"} and name in {"MLP", "GradientBoosting", "HistGradientBoosting"}:
                pred = pd.DataFrame(
                    {
                        "sample_id": np.arange(len(y_all)),
                        "dataset": "ctt",
                        "setting": setting,
                        "feature_protocol": protocol,
                        "model": name,
                        "label": y_all,
                        "score": score,
                        "prediction_val_f1_threshold": (score >= thresholds[name]).astype(int),
                        "threshold_val_f1": thresholds[name],
                        "seed": seed,
                        "negative_protocol": neg_protocol,
                    }
                )
                pred.to_csv(PREDS / f"test04_{protocol}_{name}_{neg_protocol}_seed{seed}_scores.csv", index=False)
    return rows


def dataset_version_audit() -> pd.DataFrame:
    folders = {"train": "train_01", **TEST_FOLDERS}
    rows = []
    for setting, folder in folders.items():
        files = sorted((RAW_ROOT / folder).glob("*.csv"))
        total = pos = 0
        attacks: set[str] = set()
        for path in files:
            df = pd.read_csv(path, usecols=["attack"])
            total += len(df)
            pos += int(df["attack"].astype(int).clip(0, 1).sum())
            if path.stem.startswith("attack-free") or pos == 0:
                pass
            attacks.add(path.stem.rsplit("-", 1)[0])
        rows.append(
            {
                "dataset_version": "local can-train-and-test set_01",
                "source_path": str(RAW_ROOT / folder),
                "setting": setting,
                "num_files": len(files),
                "num_samples": total,
                "num_positive": pos,
                "num_negative": total - pos,
                "attack_types": ";".join(sorted(attacks)),
                "vehicles": "unknown_from_directory_name",
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "dataset_version_audit.csv", index=False)
    (OUT / "dataset_version_audit.md").write_text(
        "# Dataset Version Audit\n\n"
        "The local data root is `data/raw/can-train-and-test/set_01`. No separate `can-train-and-test-v1.5` marker, metadata file, or version manifest was found in the local tree. Therefore exact equivalence to public v1.5 cannot be proven from local files alone.\n\n"
        f"```csv\n{out.to_csv(index=False)}```\n",
        encoding="utf-8",
    )
    return out


def feature_protocol_manifest() -> pd.DataFrame:
    rows = [
        {
            "feature_protocol": p.name,
            "columns": ";".join(p.columns),
            "safe_can_feature_set": p.safe,
            "normalization": "StandardScaler fit on train split only",
            "notes": p.notes,
        }
        for p in PROTOCOLS.values()
    ]
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "feature_protocol_manifest.csv", index=False)
    (OUT / "feature_protocol_manifest.md").write_text(
        "# Feature Protocol Manifest\n\n"
        f"```csv\n{out.to_csv(index=False)}```\n",
        encoding="utf-8",
    )
    return out


def run_public_reproduction() -> pd.DataFrame:
    scenarios = [
        ("A_capped", 60_000, 30_000, [42], ["P1_public_default", "P2_no_timestamp", "P3_no_subdivision", "P4_timestamp_only", "P5_arbitration_payload_only", "P6_delta_features", "P7_public_plus_delta", "SAFE_CAN", "RISKY_PROTOCOL"]),
        ("B_2x_negative_cap", 120_000, 60_000, [42], ["P1_public_default", "P2_no_timestamp", "SAFE_CAN", "RISKY_PROTOCOL"]),
        ("C_5x_negative_cap", 300_000, 120_000, [42], ["P1_public_default", "P2_no_timestamp", "SAFE_CAN", "RISKY_PROTOCOL"]),
    ]
    core_models = ["MLP", "GradientBoosting", "HistGradientBoosting", "LogisticRegression", "GaussianNB"]
    existing = pd.read_csv(TABLES / "public_protocol_reproduction.csv") if (TABLES / "public_protocol_reproduction.csv").exists() else pd.DataFrame()
    rows: list[dict] = existing.to_dict("records") if not existing.empty else []
    out_path = TABLES / "public_protocol_reproduction.csv"

    def completed_key(protocol: str, neg_protocol: str, seed: int) -> bool:
        if existing.empty:
            return False
        sub = existing[
            existing["feature_protocol"].astype(str).eq(protocol)
            & existing["negative_protocol"].astype(str).eq(neg_protocol)
            & existing["seed"].astype(str).eq(str(seed))
            & existing["status"].astype(str).eq("completed")
        ]
        return set(sub["setting"].astype(str)) >= {"ctt_test01", "ctt_test02", "ctt_test03", "ctt_test04"}

    def flush() -> None:
        out = pd.DataFrame(rows)
        out.to_csv(out_path, index=False)
        (TABLES / "public_protocol_reproduction.tex").write_text(
            out.to_latex(index=False, escape=True, na_rep="NA", float_format=lambda x: f"{x:.4f}"),
            encoding="utf-8",
        )

    for neg_protocol, neg_cap, val_cap, seeds, protocols in scenarios:
        for seed in seeds:
            for protocol in protocols:
                if completed_key(protocol, neg_protocol, seed):
                    print(f"[public_repro] skip completed {neg_protocol} {protocol} seed={seed}", flush=True)
                    continue
                settings = ["ctt_test01", "ctt_test02", "ctt_test03", "ctt_test04"]
                models_for_protocol = core_models
                if neg_protocol != "A_capped":
                    models_for_protocol = ["MLP", "GradientBoosting", "HistGradientBoosting"]
                scaler, fitted, thresholds, fit_times, _, y_train, _, _, status = fit_models(protocol, seed, neg_cap, val_cap, models_for_protocol)
                for s in status:
                    for setting in settings:
                        rows.append({**s, "negative_protocol": neg_protocol, "setting": setting})
                rows.extend(evaluate_models(protocol, scaler, fitted, thresholds, fit_times, y_train, neg_protocol, seed, settings))
                flush()
                if neg_protocol == "A_capped" and seed == 42 and protocol in {"P1_public_default", "P2_no_timestamp", "SAFE_CAN", "RISKY_PROTOCOL"}:
                    extra = ["RandomForest", "ExtraTrees", "IsolationForest", "BIRCH", "LinearSVM", "KNN", "XGBoost", "LightGBM", "CatBoost"]
                    scaler2, fitted2, thresholds2, fit_times2, _, y_train2, _, _, status2 = fit_models(protocol, seed, neg_cap, val_cap, extra)
                    for s in status2:
                        rows.append({**s, "negative_protocol": neg_protocol, "setting": "ctt_test04"})
                    rows.extend(evaluate_models(protocol, scaler2, fitted2, thresholds2, fit_times2, y_train2, neg_protocol, seed, ["ctt_test04"]))
                    flush()
    rows.append(
        {
            "feature_protocol": "all",
            "model": "all",
            "negative_protocol": "D_chunked_full_negative",
            "seed": "",
            "setting": "all",
            "status": "documented_resource_limit",
            "notes": "train_01 has 10,603,583 negative frames; full-negative for every feature/model/protocol would require chunked streaming implementation beyond sklearn batch estimators. This row records Protocol D feasibility evidence instead of silently skipping it.",
        }
    )
    flush()
    return pd.DataFrame(rows)


def shortcut_tables(public: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    completed = public[public["status"].astype(str).eq("completed")].copy()
    test04 = completed[completed["setting"].eq("ctt_test04")].copy()
    timestamp = test04[test04["feature_protocol"].isin(["P1_public_default", "P2_no_timestamp", "P4_timestamp_only", "RISKY_PROTOCOL", "SAFE_CAN", "P7_public_plus_delta"])].copy()
    timestamp.to_csv(TABLES / "timestamp_shortcut_audit.csv", index=False)
    ablation = test04[test04["feature_protocol"].isin(["P1_public_default", "P2_no_timestamp", "P4_timestamp_only", "P6_delta_features", "P7_public_plus_delta"])].copy()
    ablation.to_csv(TABLES / "timestamp_ablation.csv", index=False)
    safe = test04[test04["feature_protocol"].isin(["SAFE_CAN", "RISKY_PROTOCOL", "P1_public_default", "P2_no_timestamp", "P7_public_plus_delta"])].copy()
    safe.to_csv(TABLES / "safe_vs_risky_features.csv", index=False)
    for name, df in [
        ("timestamp_shortcut_audit", timestamp),
        ("timestamp_ablation", ablation),
        ("safe_vs_risky_features", safe),
    ]:
        (TABLES / f"{name}.tex").write_text(df.to_latex(index=False, escape=True, na_rep="NA", float_format=lambda x: f"{x:.4f}"), encoding="utf-8")
    return timestamp, ablation, safe


def final_low_fpr_event_metrics() -> pd.DataFrame:
    rows = []
    for path in sorted(PREDS.glob("test04_*_scores.csv")):
        df = pd.read_csv(path)
        y = df["label"].astype(int).to_numpy()
        score = df["score"].astype(float).to_numpy()
        for budget in FPR_BUDGETS:
            b = recall_at_budget(y, score, budget)
            rows.append(
                {
                    "score_file": str(path),
                    "feature_protocol": df["feature_protocol"].iloc[0],
                    "model": df["model"].iloc[0],
                    "setting": "ctt_test04",
                    "threshold_type": "best_test_upper_bound",
                    "fpr_budget": budget,
                    "recall_at_fpr": b["recall"],
                    "precision_at_fpr": b["precision"],
                    "f1_at_fpr": b["f1"],
                    "actual_fpr": b["actual_fpr"],
                    "threshold": b["threshold"],
                    "num_positive": int((y == 1).sum()),
                    "num_negative": int((y == 0).sum()),
                }
            )
        pred = df["prediction_val_f1_threshold"].astype(int).to_numpy()
        starts = np.where((y == 1) & np.r_[True, y[:-1] == 0])[0]
        ends = np.r_[starts[1:], len(y)]
        hit = 0
        for s, e in zip(starts, ends):
            if (pred[s:e] == 1).any():
                hit += 1
        rows.append(
            {
                "score_file": str(path),
                "feature_protocol": df["feature_protocol"].iloc[0],
                "model": df["model"].iloc[0],
                "setting": "ctt_test04",
                "threshold_type": "validation_f1_threshold_event_approx",
                "fpr_budget": np.nan,
                "event_recall": hit / max(len(starts), 1),
                "num_events": int(len(starts)),
                "event_boundary_quality": "approximate_label_transition",
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "final_low_fpr_event_metrics.csv", index=False)
    (TABLES / "final_low_fpr_event_metrics.tex").write_text(out.to_latex(index=False, escape=True, na_rep="NA", float_format=lambda x: f"{x:.4f}"), encoding="utf-8")
    return out


def plot_bar(df: pd.DataFrame, path: Path, title: str, x_col: str = "feature_protocol", hue_col: str = "model", value_col: str = "f1", top_n: int = 12) -> None:
    plot = df[df["status"].astype(str).eq("completed")].copy() if "status" in df else df.copy()
    plot[value_col] = pd.to_numeric(plot[value_col], errors="coerce")
    plot = plot.dropna(subset=[value_col]).sort_values(value_col, ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    labels = plot[x_col].astype(str) + "\n" + plot[hue_col].astype(str)
    hatches = ["//", "\\\\", "xx", "..", "++", "--"]
    for i, (_, row) in enumerate(plot.iterrows()):
        ax.bar(i, row[value_col], color="#D9D9D9", edgecolor="black", hatch=hatches[i % len(hatches)], linewidth=0.8)
    ax.set_xticks(np.arange(len(plot)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel(value_col.upper())
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.6)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_all(public: pd.DataFrame, timestamp: pd.DataFrame, safe: pd.DataFrame, low: pd.DataFrame) -> None:
    completed = public[public["status"].astype(str).eq("completed")].copy()
    plot_bar(completed[completed["setting"].eq("ctt_test04")], FIGS / "public_protocol_reproduction.svg", "Public Protocol Test04 Reproduction")
    plot_bar(completed[completed["setting"].eq("ctt_test04")], FIGS / "paper_fig1_public_reproduction.svg", "Public Protocol Test04 Reproduction")
    plot_bar(completed[completed["setting"].eq("ctt_test04")], FIGS / "paper_fig2_feature_protocols.svg", "Feature Protocol Comparison")
    plot_bar(timestamp, FIGS / "timestamp_shortcut_audit.svg", "Timestamp Shortcut Audit")
    plot_bar(timestamp, FIGS / "paper_fig3_timestamp_shortcut.svg", "Timestamp Shortcut Audit")
    plot_bar(safe, FIGS / "safe_vs_risky_features.svg", "Safe vs Risky Features")
    plot_bar(safe, FIGS / "paper_fig4_safe_vs_risky.svg", "Safe vs Risky Features")
    plot_bar(completed[completed["setting"].eq("ctt_test04")], FIGS / "paper_fig5_test04_leaderboard.svg", "Test04 Leaderboard")
    curves = low[low["threshold_type"].eq("best_test_upper_bound")].dropna(subset=["fpr_budget", "recall_at_fpr"]).copy()
    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    for (protocol, model), group in curves.groupby(["feature_protocol", "model"]):
        if protocol not in {"P1_public_default", "P2_no_timestamp", "SAFE_CAN", "RISKY_PROTOCOL", "P7_public_plus_delta"}:
            continue
        ax.plot(group["fpr_budget"], group["recall_at_fpr"], marker="o", linewidth=1.1, label=f"{protocol}/{model}"[:34])
    ax.set_xscale("log")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("FPR budget")
    ax.set_ylabel("Recall")
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.6)
    if not curves.empty:
        ax.legend(frameon=False, fontsize=6, ncol=1)
    fig.savefig(FIGS / "final_low_fpr_event_metrics.svg", bbox_inches="tight")
    fig.savefig(FIGS / "paper_fig6_low_fpr_event.svg", bbox_inches="tight")
    plt.close(fig)


def write_reports(public: pd.DataFrame, timestamp: pd.DataFrame, safe: pd.DataFrame, low: pd.DataFrame) -> None:
    completed = public[public["status"].astype(str).eq("completed")].copy()
    test04 = completed[completed["setting"].eq("ctt_test04")].copy()
    best = test04.sort_values("f1", ascending=False).head(20)
    safe_best = safe[safe["feature_protocol"].isin(["SAFE_CAN", "P2_no_timestamp", "P6_delta_features"])].sort_values("f1", ascending=False).head(5)
    risky_best = safe[safe["feature_protocol"].isin(["RISKY_PROTOCOL", "P1_public_default", "P4_timestamp_only", "P7_public_plus_delta"])].sort_values("f1", ascending=False).head(5)
    best_f1 = float(best["f1"].max()) if not best.empty else np.nan
    safe_f1 = float(safe_best["f1"].max()) if not safe_best.empty else np.nan
    risky_f1 = float(risky_best["f1"].max()) if not risky_best.empty else np.nan
    public_repro = best_f1 >= 0.95
    safe_high = safe_f1 >= 0.8
    risky_gap = risky_f1 - safe_f1 if not (np.isnan(risky_f1) or np.isnan(safe_f1)) else np.nan

    (OUT / "public_protocol_reproduction.md").write_text(
        "# Public Protocol Reproduction\n\n"
        f"Best CT&T test04 F1 observed in this run: {best_f1:.4f}.\n\n"
        f"```csv\n{best[['feature_protocol','model','negative_protocol','seed','setting','precision','recall','f1','auroc','aupr','fpr','fnr']].to_csv(index=False)}```\n\n"
        "The table records completed rows and explicit unavailable/resource rows; no failed model is silently omitted.\n",
        encoding="utf-8",
    )
    (OUT / "timestamp_shortcut_audit.md").write_text(
        "# Timestamp Shortcut Audit\n\n"
        f"Best risky/public test04 F1: {risky_f1:.4f}. Best safe/no-raw-timestamp test04 F1: {safe_f1:.4f}. Gap: {risky_gap:.4f}.\n\n"
        "If timestamp-only or risky-protocol rows approach the best public result, raw timestamp/capture schedule is a benchmark shortcut risk. If removing timestamp keeps performance high, the signal is more likely to be CAN behavioral.\n",
        encoding="utf-8",
    )
    (OUT / "timestamp_ablation.md").write_text(
        "# Timestamp Ablation\n\n"
        "Compare P1_public_default, P2_no_timestamp, P4_timestamp_only, P6_delta_features, and P7_public_plus_delta in `tables/timestamp_ablation.csv`. Raw timestamp is unsafe if P4 or P1/P7 dominate while P2/SAFE_CAN collapse.\n",
        encoding="utf-8",
    )
    (OUT / "safe_vs_risky_features.md").write_text(
        "# Safe vs Risky Features\n\n"
        f"Best Safe-CAN family F1: {safe_f1:.4f}. Best risky/public family F1: {risky_f1:.4f}.\n\n"
        "Safe-CAN features exclude absolute timestamp, sample index, file ID, attack ratio and any post-outcome statistics. Risky rows are audit evidence, not deployable IDS inputs.\n",
        encoding="utf-8",
    )
    if safe_high:
        strategy = "A. Safe-CAN GRAIN-CAN method paper"
    elif public_repro and risky_gap > 0.2:
        strategy = "B. Shortcut-aware benchmark correction / measurement paper"
    elif not public_repro:
        strategy = "C. Continue method/protocol reproduction"
    else:
        strategy = "B. Shortcut-aware benchmark correction / measurement paper"
    (OUT / "final_security4_strategy.md").write_text(
        "# Final Security4 Strategy\n\n"
        f"**Selected direction: {strategy}.**\n\n"
        f"- Public-style best test04 F1: {best_f1:.4f}.\n"
        f"- Safe-CAN best test04 F1: {safe_f1:.4f}.\n"
        f"- Risky/public best test04 F1: {risky_f1:.4f}.\n\n"
        "Decision rule: if Safe-CAN reaches the public high result, absorb it into GRAIN-CAN. If public/risky protocols dominate but Safe-CAN does not, write the benchmark-correction story. If neither public nor shortcut reproduction is high, continue investigating data-version and preprocessing differences.\n",
        encoding="utf-8",
    )
    (OUT / "shortcut_finding_report.md").write_text(
        "# Shortcut Finding Report\n\n"
        f"Best rows:\n\n```csv\n{best[['feature_protocol','model','negative_protocol','seed','f1','auroc','aupr','fpr','fnr']].to_csv(index=False)}```\n\n"
        f"Safe best:\n\n```csv\n{safe_best[['feature_protocol','model','negative_protocol','seed','f1','auroc','aupr','fpr','fnr']].to_csv(index=False)}```\n\n"
        f"Risky/public best:\n\n```csv\n{risky_best[['feature_protocol','model','negative_protocol','seed','f1','auroc','aupr','fpr','fnr']].to_csv(index=False)}```\n",
        encoding="utf-8",
    )
    (OUT / "final_low_fpr_event_metrics.md").write_text(
        "# Final Low-FPR And Event Metrics\n\n"
        "Low-FPR rows are computed from saved test04 score dumps and marked as best-test upper bounds. Event rows use approximate label-transition events from the prediction order. Formal deployment evidence still requires validation-budget thresholds and official event boundaries.\n\n"
        f"```csv\n{low.head(40).to_csv(index=False)}```\n",
        encoding="utf-8",
    )
    (OUT / "recommended_next_paper_claims.md").write_text(
        "# Recommended Next Paper Claims\n\n"
        "- Claim only the highest result that is supported by Safe-CAN or explicitly labeled shortcut-audit evidence.\n"
        "- If raw timestamp/risky protocol dominates, claim benchmark artifact discovery and corrected protocol, not unknown-attack IDS breakthrough.\n"
        "- If Safe-CAN reaches high test04 F1 with low-FPR/event evidence, claim feature-preserving Safe-CAN GRAIN-CAN as the main method.\n",
        encoding="utf-8",
    )
    (OUT / "unsafe_claims_do_not_write.md").write_text(
        "# Unsafe Claims Do Not Write\n\n"
        "- Do not claim public 0.998 test04 is real generalization unless Safe-CAN reproduces it.\n"
        "- Do not use timestamp-only, file-id, vehicle-id, or sample-index models as deployable IDS methods.\n"
        "- Do not present best-test low-FPR thresholds as formal validation-threshold deployment results.\n"
        "- Do not claim CT&T-v1.5 equivalence; the local tree has no v1.5 manifest.\n"
        "- Do not claim event-level deployment evidence is official; boundaries are approximate unless official metadata is supplied.\n",
        encoding="utf-8",
    )


def write_inventory() -> None:
    files = sorted(str(p) for p in OUT.rglob("*") if p.is_file())
    (OUT / "inventory.txt").write_text("\n".join(files) + "\n", encoding="utf-8")


def main() -> None:
    setup()
    (OUT / "input_inventory.txt").write_text(
        "\n".join(str(p) for root in [ROOT / "results/final_grain_can", ROOT / "results/granularity_shift", ROOT / "results/protocol_rescue"] if root.exists() for p in sorted(root.rglob("*")) if p.is_file()) + "\n",
        encoding="utf-8",
    )
    dataset_version_audit()
    feature_protocol_manifest()
    public = run_public_reproduction()
    timestamp, _, safe = shortcut_tables(public)
    low = final_low_fpr_event_metrics()
    plot_all(public, timestamp, safe, low)
    write_reports(public, timestamp, safe, low)
    write_inventory()
    print("[test04_public_reproduction] done")


if __name__ == "__main__":
    main()
