from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(".")
OUT = ROOT / "results/exact_public_reproduction"
TABLES = OUT / "tables"
FIGS = OUT / "figures"
LOGS = OUT / "logs"
PREDS = OUT / "predictions"
AUDITS = OUT / "audits"
MANIFESTS = OUT / "manifests"
CONFIGS = OUT / "configs"

EXPECTED_V15_SET01_TOTAL = 55_582_992
EXPECTED_V15_SET01_TRAIN = 11_460_705

SUBSET_ALIASES = {
    "train_01": ["train_01", "train_01_attack_free"],
    "train_02": ["train_02", "train_02_with_attacks"],
    "test01": ["test_01_known_vehicle_known_attack", "test01"],
    "test02": ["test_02_unknown_vehicle_known_attack", "test02"],
    "test03": ["test_03_known_vehicle_unknown_attack", "test03"],
    "test04": ["test_04_unknown_vehicle_unknown_attack", "test04"],
    "test05": ["test_05_suppress", "test05"],
    "test06": ["test_06_masquerade", "test06"],
}


def setup() -> None:
    for p in [TABLES, FIGS, LOGS, PREDS, AUDITS, MANIFESTS, CONFIGS]:
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


def write_table(df: pd.DataFrame, name: str) -> None:
    df.to_csv(TABLES / f"{name}.csv", index=False)
    (TABLES / f"{name}.tex").write_text(
        df.to_latex(index=False, escape=True, na_rep="NA", float_format=lambda x: f"{x:.4f}"),
        encoding="utf-8",
    )


def short_hash(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for p in sorted(paths):
        try:
            st = p.stat()
            h.update(str(p).encode("utf-8"))
            h.update(str(st.st_size).encode("utf-8"))
        except OSError:
            continue
    return h.hexdigest()[:16]


def md5_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def find_subset_dir(root: Path, key: str) -> Path | None:
    for alias in SUBSET_ALIASES[key]:
        exact = root / alias
        if exact.is_dir():
            return exact
        hits = [p for p in root.rglob(alias) if p.is_dir()]
        if hits:
            return sorted(hits, key=lambda p: len(str(p)))[0]
    return None


def candidate_roots() -> list[Path]:
    roots: list[Path] = []
    for env_name in ["CTT_ORIGINAL_ROOT", "CTT_V15_ROOT"]:
        value = os.environ.get(env_name)
        if value:
            roots.append(Path(value).expanduser())
    search_roots = [ROOT / "data/raw", ROOT / "data/external", Path.home() / "datasets"]
    names = ["can-train-and-test", "can-train-and-test-v1.5", "set_01"]
    for base in search_roots:
        if not base.exists():
            continue
        for name in names:
            for p in base.rglob(name):
                if p.is_dir():
                    roots.append(p)
    # Include parent if the hit is set_01.
    expanded = []
    for p in roots:
        expanded.append(p)
        if p.name == "set_01":
            expanded.append(p.parent)
    uniq = []
    seen = set()
    for p in expanded:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            uniq.append(p)
    return uniq


def count_csv_rows(path: Path) -> tuple[int, list[str], dict[str, int], str, str, int, str]:
    try:
        header = pd.read_csv(path, nrows=0)
        columns = list(header.columns)
    except Exception:
        return 0, [], {}, "NA", "NA", 0, "unreadable"
    label_col = "attack" if "attack" in columns else ("label" if "label" in columns else None)
    counts: dict[str, int] = {}
    min_ts = "NA"
    max_ts = "NA"
    can_ids: set[str] = set()
    payload_format = "NA"
    rows = 0
    usecols = [c for c in ["timestamp", "arbitration_id", "data_field", label_col] if c]
    try:
        for chunk in pd.read_csv(path, usecols=usecols, chunksize=500_000):
            rows += len(chunk)
            if label_col and label_col in chunk:
                vc = chunk[label_col].astype(str).value_counts()
                for k, v in vc.items():
                    counts[k] = counts.get(k, 0) + int(v)
            if "timestamp" in chunk and len(chunk):
                vals = pd.to_numeric(chunk["timestamp"], errors="coerce")
                mn = vals.min()
                mx = vals.max()
                if pd.notna(mn):
                    min_ts = str(mn) if min_ts == "NA" else str(min(float(min_ts), float(mn)))
                if pd.notna(mx):
                    max_ts = str(mx) if max_ts == "NA" else str(max(float(max_ts), float(mx)))
            if "arbitration_id" in chunk:
                can_ids.update(chunk["arbitration_id"].dropna().astype(str).unique().tolist()[:5000])
            if "data_field" in chunk and payload_format == "NA":
                sample = chunk["data_field"].dropna().astype(str).head(10).tolist()
                if sample:
                    payload_format = "hex_string_no_spaces" if all(" " not in s for s in sample) else "hex_string_with_spaces"
    except Exception as exc:
        payload_format = f"read_error:{type(exc).__name__}"
    return rows, columns, counts, min_ts, max_ts, len(can_ids), payload_format


def reproduction_targets() -> pd.DataFrame:
    rows = [
        {
            "target_id": "A",
            "paper": "Lampe & Meng 2024 / can-train-and-test original paper",
            "dataset_version": "can-train-and-test original",
            "repository": "brooke-lampe/can-train-and-test Bitbucket / original dataset release",
            "subset": "Sub-dataset #1 / set_01",
            "test_setting": "Testing subset #4 / unknown vehicle + unknown attack",
            "expected_model": "BIRCH; GradientBoosting; LogisticRegression; MLP; IsolationForest",
            "expected_f1": "reported near 0.998 for selected Table-13-style benchmark rows per task statement",
            "expected_features": "timestamp; arbitration ID; data field / payload variants",
            "expected_metric": "F1, exact positive class definition must be verified",
        },
        {
            "target_id": "B",
            "paper": "can-sleuth 2025 / CT&T-v1.5 benchmark",
            "dataset_version": "can-train-and-test-v1.5",
            "repository": "can-sleuth / can-train-and-test-v1.5 release",
            "subset": "set_01",
            "test_setting": "test04",
            "expected_model": "MLP",
            "expected_f1": "0.9981 per task statement",
            "expected_features": "timestamp; arbitration ID; data field subdivided/intact variants",
            "expected_metric": "F1, exact averaging/positive-label definition must be verified",
        },
    ]
    df = pd.DataFrame(rows)
    write_table(df, "reproduction_targets")
    (OUT / "reproduction_targets.md").write_text(
        "# Reproduction Targets\n\n"
        "This task locks two separate targets. Exact reproduction is only valid if the dataset version, file manifest, feature protocol, model parameters and metric definition all align.\n\n"
        f"```csv\n{df.to_csv(index=False)}```\n",
        encoding="utf-8",
    )
    return df


def dataset_fingerprint() -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = candidate_roots()
    if not candidates:
        candidates = [ROOT / "data/raw/can-train-and-test"]
    rows = []
    manifest_rows = []
    for root in candidates:
        csvs = sorted(root.rglob("*.csv")) if root.exists() else []
        subsets = {key: find_subset_dir(root, key) if root.exists() else None for key in SUBSET_ALIASES}
        subset_counts: dict[str, int] = {}
        subset_cols: dict[str, str] = {}
        subset_labels: dict[str, str] = {}
        for key, folder in subsets.items():
            if folder is None:
                subset_counts[key] = 0
                continue
            total = 0
            label_total: dict[str, int] = {}
            cols_seen = ""
            for file in sorted(folder.glob("*.csv")):
                n, cols, labels, min_ts, max_ts, cid_count, payload_format = count_csv_rows(file)
                total += n
                cols_seen = ";".join(cols) if cols and not cols_seen else cols_seen
                for k, v in labels.items():
                    label_total[k] = label_total.get(k, 0) + v
                attack_type = file.stem.rsplit("-", 1)[0]
                manifest_rows.append(
                    {
                        "dataset_version": root.name,
                        "subset": key,
                        "file_path": str(file),
                        "file_name": file.name,
                        "file_size_bytes": file.stat().st_size,
                        "row_count": n,
                        "columns": ";".join(cols),
                        "label_distribution": json.dumps(labels, sort_keys=True),
                        "attack_type_distribution": json.dumps({attack_type: int(labels.get("1", 0) + labels.get("True", 0))}, sort_keys=True),
                        "min_timestamp": min_ts,
                        "max_timestamp": max_ts,
                        "can_id_count": cid_count,
                        "payload_column_format": payload_format,
                        "hash_md5": md5_file(file),
                    }
                )
            subset_counts[key] = total
            subset_cols[key] = cols_seen
            subset_labels[key] = json.dumps(label_total, sort_keys=True)
        total_samples = sum(subset_counts.values())
        has_v15_counts = total_samples == EXPECTED_V15_SET01_TOTAL and subset_counts.get("train_01", 0) == EXPECTED_V15_SET01_TRAIN
        has_v15_dirs = subsets["test05"] is not None and subsets["test06"] is not None and subsets["train_02"] is not None
        has_original_sets = all((root / f"set_{i:02d}").is_dir() for i in range(1, 5)) if root.exists() else False
        status = "aligned_v15" if has_v15_counts and has_v15_dirs else "not_aligned"
        is_public_set01 = (
            root.name == "set_01"
            and root.parent.name == "can-train-and-test"
            and subsets["train_01"] is not None
            and subsets["test04"] is not None
        )
        if status == "not_aligned" and root.name == "can-train-and-test" and has_original_sets and subsets["train_01"] and subsets["test04"]:
            status = "aligned_original_public_bitbucket"
        elif status == "not_aligned" and is_public_set01:
            status = "aligned_original_public_bitbucket_set01"
        elif root.exists() and subsets["train_01"] and subsets["test04"]:
            status = "candidate_original_set01"
        rows.append(
            {
                "dataset_candidate": root.name,
                "path": str(root),
                "num_files": len(csvs),
                "total_size_bytes": sum(p.stat().st_size for p in csvs),
                "has_train_01": subsets["train_01"] is not None,
                "has_train_02": subsets["train_02"] is not None,
                "has_test01": subsets["test01"] is not None,
                "has_test02": subsets["test02"] is not None,
                "has_test03": subsets["test03"] is not None,
                "has_test04": subsets["test04"] is not None,
                "has_test05": subsets["test05"] is not None,
                "has_test06": subsets["test06"] is not None,
                "has_set_01": (root / "set_01").is_dir() if root.exists() else False,
                "has_set_02": (root / "set_02").is_dir() if root.exists() else False,
                "has_set_03": (root / "set_03").is_dir() if root.exists() else False,
                "has_set_04": (root / "set_04").is_dir() if root.exists() else False,
                "has_data_extended_xlsx": any(p.name == "data-extended.xlsx" for p in root.rglob("*")) if root.exists() else False,
                "has_can_ml": any("can-ml" in str(p).lower() for p in root.rglob("*")) if root.exists() else False,
                "file_list_hash": short_hash(csvs),
                "sample_count_train": subset_counts.get("train_01", 0) + subset_counts.get("train_02", 0),
                "sample_count_test01": subset_counts.get("test01", 0),
                "sample_count_test02": subset_counts.get("test02", 0),
                "sample_count_test03": subset_counts.get("test03", 0),
                "sample_count_test04": subset_counts.get("test04", 0),
                "sample_count_total_set01_visible": total_samples,
                "expected_v15_total_samples": EXPECTED_V15_SET01_TOTAL,
                "expected_v15_train_samples": EXPECTED_V15_SET01_TRAIN,
                "columns_train": subset_cols.get("train_01", ""),
                "columns_test04": subset_cols.get("test04", ""),
                "label_values": subset_labels.get("train_01", ""),
                "attack_types_train": ";".join(sorted({r["file_name"].rsplit("-", 1)[0] for r in manifest_rows if r["dataset_version"] == root.name and r["subset"] == "train_01"})),
                "attack_types_test04": ";".join(sorted({r["file_name"].rsplit("-", 1)[0] for r in manifest_rows if r["dataset_version"] == root.name and r["subset"] == "test04"})),
                "vehicle_or_source_fields": "not_present_in_csv_columns",
                "status": status,
            }
        )
    fp = pd.DataFrame(rows)
    manifest = pd.DataFrame(manifest_rows)
    write_table(fp, "dataset_version_fingerprint")
    manifest.to_csv(MANIFESTS / "set01_file_manifest.csv", index=False)
    (OUT / "dataset_version_fingerprint.md").write_text(
        "# Dataset Version Fingerprint\n\n"
        "Exact alignment requires matching the public dataset version and file manifest. The task-provided can-sleuth v1.5 anchors are set_01 total samples = 55,582,992 and training samples = 11,460,705.\n\n"
        f"```csv\n{fp.to_csv(index=False)}```\n",
        encoding="utf-8",
    )
    (OUT / "set01_file_manifest.md").write_text(
        "# Set01 File Manifest\n\n"
        f"Rows: {len(manifest)}. File-level hashes, row counts, labels, timestamp ranges and payload formats are saved in `manifests/set01_file_manifest.csv`.\n",
        encoding="utf-8",
    )
    if not (fp["status"].eq("aligned_v15").any()):
        (OUT / "missing_dataset_version.md").write_text(
            "# Missing Dataset Version\n\n"
            "No local candidate matches the task-provided can-train-and-test-v1.5 set_01 anchors: total samples 55,582,992 and training samples 11,460,705 with train_02/test05/test06 support.\n\n"
            "The public original Bitbucket repository is handled separately as Target A when the full `set_01`-`set_04` tree is present.\n\n"
            "Current local data found:\n\n"
            f"```csv\n{fp.to_csv(index=False)}```\n\n"
            "To continue exact reproduction, provide the original can-train-and-test release and/or can-train-and-test-v1.5 under one of:\n"
            "- `CTT_ORIGINAL_ROOT=/path/to/can-train-and-test`\n"
            "- `CTT_V15_ROOT=/path/to/can-train-and-test-v1.5`\n"
            "- `data/external/can-train-and-test-v1.5`\n\n"
            "Then rerun: `.venv/bin/python -m cmf_can.analysis.exact_public_reproduction`.\n",
            encoding="utf-8",
        )
    return fp, manifest


def protocol_manifests() -> tuple[pd.DataFrame, pd.DataFrame]:
    features = [
        ("P1_public_default_subdivided", "timestamp;arbitration_id;byte_1;byte_2;byte_3;byte_4;byte_5;byte_6;byte_7;byte_8", True, True, True, False, False, False, "fit train only", "numeric scaling", "public default candidate"),
        ("P2_public_no_timestamp_subdivided", "arbitration_id;byte_1;byte_2;byte_3;byte_4;byte_5;byte_6;byte_7;byte_8", False, True, True, False, False, False, "fit train only", "numeric scaling", "raw timestamp ablation"),
        ("P3_public_intact_datafield", "timestamp;arbitration_id;data_field_as_single_int_or_hex_numeric", True, True, False, True, False, False, "fit train only", "numeric scaling", "intact data field candidate"),
        ("P4_public_intact_no_timestamp", "arbitration_id;data_field_as_single_int_or_hex_numeric", False, True, False, True, False, False, "fit train only", "numeric scaling", "intact no timestamp"),
        ("P5_timestamp_only", "timestamp", True, False, False, False, False, False, "fit train only", "numeric scaling", "shortcut sanity"),
        ("P6_arbitration_only", "arbitration_id", False, True, False, False, False, False, "fit train only", "numeric scaling", "ID-only sanity"),
        ("P7_payload_bytes_only", "byte_1;byte_2;byte_3;byte_4;byte_5;byte_6;byte_7;byte_8", False, False, True, False, False, False, "fit train only", "numeric scaling", "payload-only sanity"),
        ("P8_current_safe_can", "can_id;payload bytes;safe causal deltas;payload stats", False, True, True, False, True, False, "fit train only", "numeric scaling", "current safe feature set"),
        ("P9_public_plus_safe_can", "public_default + safe causal deltas", True, True, True, False, True, False, "fit train only", "numeric scaling", "public plus safe"),
    ]
    fdf = pd.DataFrame(
        features,
        columns=[
            "protocol",
            "feature_names",
            "uses_raw_timestamp",
            "uses_arbitration_id",
            "uses_subdivided_data_bytes",
            "uses_intact_data_field",
            "uses_safe_deltas",
            "uses_future_info",
            "normalization",
            "encoding",
            "notes",
        ],
    )
    write_table(fdf, "feature_protocol_manifest")
    (OUT / "feature_protocol_manifest.md").write_text("# Feature Protocol Manifest\n\n```csv\n" + fdf.to_csv(index=False) + "```\n", encoding="utf-8")
    models = [
        ("BIRCH", "default_sklearn", {"threshold": 0.5, "n_clusters": None}, "validation_majority_or_distance_threshold", "validation_f1", True, "cluster-to-label mapping must not use test labels"),
        ("GradientBoosting", "default_sklearn", {}, "not_applicable", "validation_f1", True, "sklearn default unless paper parameters available"),
        ("LogisticRegression", "default_sklearn", {"max_iter": 100}, "not_applicable", "validation_f1", True, "sklearn default"),
        ("MLP", "default_sklearn", {"hidden_layer_sizes": [100], "activation": "relu", "solver": "adam", "max_iter": 200, "random_state": 42}, "not_applicable", "validation_f1", True, "paper exact parameters unavailable locally"),
        ("IsolationForest", "default_sklearn", {}, "distance_or_score_threshold", "validation_f1", True, "unsupervised score threshold selected on validation"),
        ("GaussianNB", "default_sklearn", {}, "not_applicable", "validation_f1", True, "sklearn default"),
        ("RandomForest", "default_sklearn", {}, "not_applicable", "validation_f1", True, "sklearn default"),
        ("ExtraTrees", "default_sklearn", {}, "not_applicable", "validation_f1", True, "sklearn default"),
        ("KNN", "default_sklearn", {}, "not_applicable", "validation_f1", True, "full CT&T scale feasibility uncertain"),
        ("LinearSVM", "default_sklearn", {}, "not_applicable", "validation_f1", True, "decision score threshold on validation"),
        ("MiniBatchKMeans", "default_sklearn", {}, "validation_majority_label", "validation_f1", True, "cluster mapping must use validation/train only"),
    ]
    mdf = pd.DataFrame(
        [
            {
                "model": m,
                "parameter_set": ps,
                "parameters_json": json.dumps(params, sort_keys=True),
                "label_mapping_rule": rule,
                "threshold_rule": thr,
                "uses_validation": val,
                "notes": notes,
            }
            for m, ps, params, rule, thr, val, notes in models
        ]
    )
    write_table(mdf, "model_protocol_manifest")
    (OUT / "model_protocol_manifest.md").write_text("# Model Protocol Manifest\n\n```csv\n" + mdf.to_csv(index=False) + "```\n", encoding="utf-8")
    return fdf, mdf


def metric_sanity() -> pd.DataFrame:
    source = ROOT / "results/test04_public_reproduction/tables/public_protocol_reproduction.csv"
    rows = []
    if source.exists():
        df = pd.read_csv(source)
        sub = df[(df["setting"].eq("ctt_test04")) & (df["status"].eq("completed"))].copy()
        for _, r in sub.sort_values("f1", ascending=False).head(20).iterrows():
            # Accuracy/normal-positive F1 cannot be recomputed without confusion matrix.
            rows.append(
                {
                    "source": str(source),
                    "feature_protocol": r.get("feature_protocol"),
                    "model": r.get("model"),
                    "binary_f1_attack_positive": r.get("f1"),
                    "binary_f1_normal_positive": "not_recomputable_without_confusion_matrix",
                    "macro_f1": "not_available_in_prior_table",
                    "weighted_f1": "not_available_in_prior_table",
                    "accuracy": "not_available_in_prior_table",
                    "precision_attack_positive": r.get("precision"),
                    "recall_attack_positive": r.get("recall"),
                    "confusion_matrix": "not_available_in_prior_table",
                    "interpretation": "Prior local reproduction did not reach 0.95 attack-positive F1.",
                }
            )
    if not rows:
        rows.append(
            {
                "source": "none",
                "feature_protocol": "NA",
                "model": "NA",
                "binary_f1_attack_positive": "NA",
                "binary_f1_normal_positive": "not_evaluated",
                "macro_f1": "not_evaluated",
                "weighted_f1": "not_evaluated",
                "accuracy": "not_evaluated",
                "precision_attack_positive": "NA",
                "recall_attack_positive": "NA",
                "confusion_matrix": "NA",
                "interpretation": "Exact reproduction sweep blocked before model evaluation.",
            }
        )
    out = pd.DataFrame(rows)
    write_table(out, "metric_definition_sanity")
    (OUT / "metric_definition_sanity.md").write_text(
        "# Metric Definition Sanity\n\n"
        "Exact metric sanity requires confusion matrices from the exact public protocol. Existing local reproduction rows only support attack-positive F1 checks; they do not prove whether public 0.998 was accuracy, normal-positive F1, or attack-positive F1.\n\n"
        f"```csv\n{out.to_csv(index=False)}```\n",
        encoding="utf-8",
    )
    return out


def exact_sweep(fp: pd.DataFrame) -> pd.DataFrame:
    aligned_v15 = bool(fp["status"].eq("aligned_v15").any())
    aligned_original = bool(fp["status"].isin(["aligned_original_public_bitbucket", "aligned_original_public_bitbucket_set01"]).any())
    rows = []
    if aligned_original:
        source = ROOT / "results/test04_public_reproduction/tables/public_protocol_reproduction.csv"
        if source.exists():
            prior = pd.read_csv(source)
            sub = prior[
                prior["setting"].astype(str).eq("ctt_test04")
                & prior["status"].astype(str).eq("completed")
            ].copy()
            for _, r in sub.sort_values("f1", ascending=False).head(80).iterrows():
                rows.append(
                    {
                        "target_id": "A",
                        "dataset_version": "can-train-and-test original Bitbucket",
                        "feature_protocol": r.get("feature_protocol"),
                        "model": r.get("model"),
                        "parameter_set": "sklearn_default_or_local_grid",
                        "setting": "test04",
                        "binary_f1_attack_positive": r.get("f1"),
                        "binary_f1_normal_positive": np.nan,
                        "macro_f1": r.get("macro_f1", np.nan),
                        "weighted_f1": np.nan,
                        "accuracy": np.nan,
                        "precision_attack_positive": r.get("precision"),
                        "recall_attack_positive": r.get("recall"),
                        "auroc": r.get("auroc", np.nan),
                        "aupr": r.get("aupr", np.nan),
                        "confusion_matrix": "not_saved_in_prior_sweep",
                        "negative_protocol": r.get("negative_protocol", "unknown"),
                        "seed": r.get("seed", "unknown"),
                        "status": "completed_sampling_approximation",
                        "blocking_layer": "none_for_Target_A_data; model_sampling_approximation_remains",
                        "notes": "Target A original Bitbucket data tree is present. This row uses the completed public-protocol sweep on set_01; it is not v1.5 and is not full-negative unless negative_protocol says so.",
                    }
                )
        else:
            rows.append(
                {
                    "target_id": "A",
                    "dataset_version": "can-train-and-test original Bitbucket",
                    "feature_protocol": "not_run",
                    "model": "not_run",
                    "parameter_set": "not_run",
                    "setting": "test04",
                    "binary_f1_attack_positive": np.nan,
                    "binary_f1_normal_positive": np.nan,
                    "macro_f1": np.nan,
                    "weighted_f1": np.nan,
                    "accuracy": np.nan,
                    "precision_attack_positive": np.nan,
                    "recall_attack_positive": np.nan,
                    "auroc": np.nan,
                    "aupr": np.nan,
                    "confusion_matrix": "NA",
                    "negative_protocol": "NA",
                    "seed": "NA",
                    "status": "blocked_missing_prior_sweep",
                    "blocking_layer": "model_evaluation",
                    "notes": "Original public data is present but the public-protocol sweep table is missing.",
                }
            )
    if not aligned_v15:
        rows.append(
            {
                "target_id": "B",
                "dataset_version": "can-train-and-test-v1.5",
                "feature_protocol": "not_run",
                "model": "not_run",
                "parameter_set": "not_run",
                "setting": "test04",
                "binary_f1_attack_positive": np.nan,
                "binary_f1_normal_positive": np.nan,
                "macro_f1": np.nan,
                "weighted_f1": np.nan,
                "accuracy": np.nan,
                "precision_attack_positive": np.nan,
                "recall_attack_positive": np.nan,
                "auroc": np.nan,
                "aupr": np.nan,
                "confusion_matrix": "NA",
                "status": "blocked_dataset_not_aligned",
                "blocking_layer": "dataset_version/file_manifest",
                "notes": "No local candidate matches can-train-and-test-v1.5 anchors; cannot run Target B/can-sleuth exact reproduction without that data version.",
            }
        )
    out = pd.DataFrame(rows)
    write_table(out, "exact_reproduction_sweep")
    (OUT / "exact_reproduction_sweep.md").write_text(
        "# Exact Reproduction Sweep\n\n"
        "Target A uses the public original Bitbucket repository when the full `set_01`-`set_04` tree is present. Target B remains blocked unless a local candidate matches the can-train-and-test-v1.5 anchors.\n\n"
        f"```csv\n{out.to_csv(index=False)}```\n",
        encoding="utf-8",
    )
    return out


def reports(fp: pd.DataFrame, sweep: pd.DataFrame) -> None:
    aligned_v15 = bool(fp["status"].eq("aligned_v15").any())
    aligned_original = bool(fp["status"].isin(["aligned_original_public_bitbucket", "aligned_original_public_bitbucket_set01"]).any())
    reproduced_high = bool(pd.to_numeric(sweep.get("binary_f1_attack_positive", pd.Series(dtype=float)), errors="coerce").fillna(0).ge(0.95).any())
    blocker = not reproduced_high
    if blocker:
        (OUT / "reproduction_blocker_report.md").write_text(
            "# Reproduction Blocker Report\n\n"
            "Exact reproduction did not reach test04 attack-positive F1 >= 0.95.\n\n"
            f"1. Target A original Bitbucket data tree present: {'yes' if aligned_original else 'no'}.\n"
            f"2. Target B v1.5 data version aligned: {'yes' if aligned_v15 else 'no'}.\n"
            "3. Official feature table / can-sleuth preprocessing code: not present locally.\n"
            "4. Current Target A sweep rows are marked `completed_sampling_approximation` unless a full-negative protocol is explicitly present.\n"
            "5. Required next input for Target B: the exact can-train-and-test-v1.5 release or can-sleuth feature table/code path under `CTT_V15_ROOT`.\n\n"
            "This file is an execution blocker only for reproducing the public high score, not a reason to stop downloading or auditing data.\n",
            encoding="utf-8",
        )
    (OUT / "final_reproduction_verdict.md").write_text(
        "# Final Reproduction Verdict\n\n"
        f"1. Target A original Bitbucket data obtained: {'yes' if aligned_original else 'no'}.\n"
        f"2. Target B can-train-and-test-v1.5 data obtained: {'yes' if aligned_v15 else 'no'}.\n"
        f"3. test04≈0.998 reproduced in available sweeps: {'yes' if reproduced_high else 'no'}.\n"
        "4. If not reproduced, the current strongest available evidence is in `tables/exact_reproduction_sweep.csv`; Target B remains impossible to execute without the exact v1.5 source tree.\n"
        "5. Existing local data can be used for Target A original Bitbucket alignment after the full public repository download, but not for Target B/can-sleuth v1.5 claims unless the v1.5 fingerprint matches.\n"
        "6. Next step is to run full-negative/model-parameter exact sweeps on Target A and add Target B immediately if a v1.5 source is found or provided.\n",
        encoding="utf-8",
    )
    (OUT / "unsafe_claims_do_not_write.md").write_text(
        "# Unsafe Claims Do Not Write\n\n"
        "- Do not claim our model fails to match public 0.998 before dataset/protocol alignment is proven.\n"
        "- Do not claim the public result is wrong without exact reproduction.\n"
        "- Do not claim current set_01 equals can-train-and-test-v1.5 unless fingerprint matches.\n"
        "- Do not claim timestamp shortcut explains public 0.998 unless exact public data confirms it.\n"
        "- Do not use local approximate reproduction as Table 13 / can-sleuth exact reproduction.\n",
        encoding="utf-8",
    )


def plots(fp: pd.DataFrame, fdf: pd.DataFrame, metrics: pd.DataFrame, sweep: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    labels = fp["dataset_candidate"].astype(str)
    values = fp["sample_count_total_set01_visible"].astype(float)
    ax.bar(np.arange(len(fp)), values, color="#D9D9D9", edgecolor="black", hatch="//")
    ax.axhline(EXPECTED_V15_SET01_TOTAL, color="black", linestyle="--", linewidth=1.0, label="v1.5 expected total")
    ax.set_xticks(np.arange(len(fp)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Visible set_01 samples")
    ax.set_title("Dataset Alignment")
    ax.legend(frameon=False, fontsize=7)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.6)
    fig.savefig(FIGS / "paper_fig1_dataset_alignment.svg", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    flags = ["uses_raw_timestamp", "uses_subdivided_data_bytes", "uses_intact_data_field", "uses_safe_deltas"]
    mat = fdf[flags].astype(int).to_numpy()
    ax.imshow(mat, cmap="Greys", aspect="auto", vmin=0, vmax=1)
    ax.set_yticks(np.arange(len(fdf)))
    ax.set_yticklabels(fdf["protocol"], fontsize=6)
    ax.set_xticks(np.arange(len(flags)))
    ax.set_xticklabels(flags, rotation=35, ha="right", fontsize=7)
    ax.set_title("Feature Protocol Gap")
    fig.savefig(FIGS / "paper_fig2_feature_protocol_gap.svg", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    vals = pd.to_numeric(metrics["binary_f1_attack_positive"], errors="coerce").dropna().head(10)
    if len(vals):
        ax.bar(np.arange(len(vals)), vals, color="#D9D9D9", edgecolor="black", hatch="xx")
    else:
        ax.text(0.5, 0.5, "Exact metric sanity blocked", ha="center", va="center")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Attack-positive F1")
    ax.set_title("Metric Sanity")
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.6)
    fig.savefig(FIGS / "paper_fig3_metric_sanity.svg", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    sweep_vals = sweep.copy()
    sweep_vals["binary_f1_attack_positive"] = pd.to_numeric(sweep_vals.get("binary_f1_attack_positive"), errors="coerce")
    sweep_vals = sweep_vals.dropna(subset=["binary_f1_attack_positive"]).sort_values("binary_f1_attack_positive", ascending=False).head(10)
    if len(sweep_vals):
        labels = (sweep_vals["feature_protocol"].astype(str) + "\n" + sweep_vals["model"].astype(str)).tolist()
        ax.bar(np.arange(len(sweep_vals)), sweep_vals["binary_f1_attack_positive"], color="#D9D9D9", edgecolor="black", hatch="--")
        ax.axhline(0.95, color="black", linestyle=":", linewidth=1.0, label="0.95 reproduction target")
        ax.set_xticks(np.arange(len(sweep_vals)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
        ax.legend(frameon=False, fontsize=7)
    else:
        ax.bar([0], [0], color="#D9D9D9", edgecolor="black", hatch="--")
        ax.set_xticks([0])
        ax.set_xticklabels(["blocked"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Exact reproduced F1")
    ax.set_title("Reproduction Sweep")
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.6)
    fig.savefig(FIGS / "paper_fig4_reproduction_sweep.svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    setup()
    reproduction_targets()
    fp, _ = dataset_fingerprint()
    fdf, _ = protocol_manifests()
    metrics = metric_sanity()
    sweep = exact_sweep(fp)
    reports(fp, sweep)
    plots(fp, fdf, metrics, sweep)
    (OUT / "inventory.txt").write_text("\n".join(str(p) for p in sorted(OUT.rglob("*")) if p.is_file()) + "\n", encoding="utf-8")
    print("[exact_public_reproduction] done")


if __name__ == "__main__":
    main()
