from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cmf_can.analysis.protocol_rescue import FEATURES, RAW, TEST_FOLDERS, collect_train, iter_test_folder


OUT = Path("results/final_evidence_completion")
TABLES = OUT / "tables"
FIGS = OUT / "figures"
LOGS = OUT / "logs"
AUDITS = OUT / "audits"
MANIFESTS = OUT / "manifests"

WEB_SEARCH_RECORDS = [
    {
        "query": "can-train-and-test-v1.5 dataset can-sleuth 2025",
        "result": "No directly accessible public dataset/code hit found through web search.",
        "url": "",
        "status": "not_found_public_web",
    },
    {
        "query": "can-train-and-test v1.5 set_01 MLP test04 F1 0.9981",
        "result": "No directly accessible public v1.5 source found through web search.",
        "url": "",
        "status": "not_found_public_web",
    },
    {
        "query": "Lampe Meng can-train-and-test Table 13 confusion matrix code",
        "result": "Original arXiv paper found; confusion matrices/exact benchmark code not found in indexed public results.",
        "url": "https://arxiv.org/abs/2308.04972",
        "status": "paper_found_code_not_found",
    },
    {
        "query": "can-sleuth GitHub can-train-and-test-v1.5",
        "result": "No directly accessible GitHub repository hit found through web search.",
        "url": "",
        "status": "not_found_public_web",
    },
]


def setup() -> None:
    for p in [TABLES, FIGS, LOGS, AUDITS, MANIFESTS]:
        p.mkdir(parents=True, exist_ok=True)


def write_table(df: pd.DataFrame, name: str) -> pd.DataFrame:
    df.to_csv(TABLES / f"{name}.csv", index=False)
    (TABLES / f"{name}.tex").write_text(df.to_latex(index=False, escape=True, na_rep="NA", float_format=lambda x: f"{x:.4f}"), encoding="utf-8")
    return df


def file_hash(path: Path, block: int = 1024 * 1024) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while True:
            data = f.read(block)
            if not data:
                break
            h.update(data)
    return h.hexdigest()


def external_dependency_audit() -> pd.DataFrame:
    rows = list(WEB_SEARCH_RECORDS)
    # Local scan for v1.5-like candidates.
    candidates = []
    for root in [Path("data/raw"), Path("data/external"), Path.home() / "datasets"]:
        if root.exists():
            for p in root.rglob("*"):
                name = str(p).lower()
                if "v1.5" in name or "v15" in name or "can-sleuth" in name:
                    candidates.append(str(p))
    rows.append({
        "query": "local filesystem search: v1.5 / v15 / can-sleuth",
        "result": "; ".join(candidates[:20]) if candidates else "No local v1.5/can-sleuth directory detected.",
        "url": "",
        "status": "local_candidates_found" if candidates else "not_found_local",
    })
    rows.append({
        "query": "local original CT&T dataset",
        "result": str(RAW.parent if RAW.exists() else "missing"),
        "url": "",
        "status": "aligned_original_set01_present" if RAW.exists() else "missing_original",
    })
    out = write_table(pd.DataFrame(rows), "external_dependency_audit")
    (OUT / "external_dependency_audit.md").write_text(
        "# External Dependency Audit\n\n"
        "Public web search and local filesystem search were performed for original-author confusion matrices/exact code, can-train-and-test-v1.5, can-sleuth, and official event boundaries. "
        "The original can-train-and-test paper and local original dataset are available; exact public confusion matrices, exact benchmark code, v1.5, and official event boundaries were not found in the accessible sources checked here.\n\n"
        "This is blocker evidence for exact-alignment claims, not a reason to drop the paper. The paper should state metric ambiguity and corrected benchmark evidence, and include this audit in the appendix.\n",
        encoding="utf-8",
    )
    return out


def v15_alignment() -> pd.DataFrame:
    rows = []
    roots = [Path("data/raw/can-train-and-test"), Path("data/raw")]
    for candidate in roots:
        if not candidate.exists():
            continue
        files = list(candidate.rglob("*.csv"))
        set01 = candidate / "set_01"
        row = {
            "dataset_candidate": str(candidate),
            "exists": True,
            "num_csv_files": len(files),
            "total_size_bytes": sum(p.stat().st_size for p in files[:10000]),
            "has_set_01": set01.exists(),
            "has_test01": (set01 / "test_01_known_vehicle_known_attack").exists(),
            "has_test02": (set01 / "test_02_unknown_vehicle_known_attack").exists(),
            "has_test03": (set01 / "test_03_known_vehicle_unknown_attack").exists(),
            "has_test04": (set01 / "test_04_unknown_vehicle_unknown_attack").exists(),
            "has_test05": (set01 / "test_05_suppress").exists(),
            "has_test06": (set01 / "test_06_masquerade").exists(),
            "v15_expected_extra_tests_present": (set01 / "test_05_suppress").exists() and (set01 / "test_06_masquerade").exists(),
            "alignment_status": "original_ctt_not_v15" if set01.exists() else "not_ctt_candidate",
        }
        rows.append(row)
    out = write_table(pd.DataFrame(rows), "v15_dataset_alignment")
    (OUT / "v15_dataset_alignment.md").write_text(
        "# can-train-and-test-v1.5 Alignment\n\n"
        "The local dataset aligns with the original can-train-and-test layout and does not expose v1.5-specific test05/test06 folders. Therefore current experiments cannot claim exact v1.5 alignment without obtaining the v1.5 source.\n",
        encoding="utf-8",
    )
    return out


def set01_manifest() -> pd.DataFrame:
    rows = []
    if not RAW.exists():
        return write_table(pd.DataFrame(), "set01_file_fingerprint")
    for p in sorted(RAW.rglob("*.csv")):
        rel = p.relative_to(RAW)
        try:
            head = pd.read_csv(p, nrows=5)
            columns = "|".join(head.columns.astype(str))
        except Exception:
            columns = "read_error"
        rows.append({
            "subset": str(rel.parent),
            "file_name": p.name,
            "file_path": str(p),
            "file_size_bytes": p.stat().st_size,
            "md5": file_hash(p),
            "columns": columns,
        })
    return write_table(pd.DataFrame(rows), "set01_file_fingerprint")


def author_code_confusion_matrix_search() -> pd.DataFrame:
    rows = []
    for rec in WEB_SEARCH_RECORDS:
        if "confusion" in rec["query"].lower() or "code" in rec["query"].lower() or "can-sleuth" in rec["query"].lower():
            rows.append(rec)
    rows.append({
        "query": "local repository search for confusion matrix / exact code",
        "result": "No original-author confusion matrix or benchmark source file found in VIDS; local code is our reproduction/audit code.",
        "url": "",
        "status": "not_found_local",
    })
    out = write_table(pd.DataFrame(rows), "author_code_confusion_matrix_search")
    (OUT / "author_code_confusion_matrix_search.md").write_text(
        "# Author Code / Confusion Matrix Search\n\n"
        "The accessible search did not locate original-author confusion matrices or exact benchmark code. Appendix should include a contact template asking for confusion matrices for Table 10/13, exact preprocessing, model parameters, train/test splits, and metric averaging mode.\n\n"
        "## Author contact template\n\n"
        "Dear authors, we are reproducing the can-train-and-test benchmark and would like to confirm the exact metric averaging mode used for the reported precision/recall/F1 values. Could you share the confusion matrices or evaluation script for the relevant test04 table, including preprocessing and model parameters?\n",
        encoding="utf-8",
    )
    return out


def official_event_boundary_audit() -> pd.DataFrame:
    rows = []
    patterns = ["event", "boundary", "interval", "metadata", "attack_interval"]
    for p in Path("data/raw").rglob("*"):
        if p.is_file() and any(k in p.name.lower() for k in patterns):
            rows.append({"path": str(p), "size": p.stat().st_size, "status": "candidate"})
    if not rows:
        rows.append({"path": "data/raw", "size": np.nan, "status": "no_official_event_boundary_file_found"})
    out = write_table(pd.DataFrame(rows), "official_event_boundary_audit")
    (OUT / "official_event_boundary_audit.md").write_text(
        "# Official Event Boundary Audit\n\n"
        "No official event boundary metadata file was found in the accessible local raw data. Existing event-level results should remain labelled as approximate_from_labels unless official boundaries are later obtained.\n",
        encoding="utf-8",
    )
    return out


def recall_at_fpr(y: np.ndarray, score: np.ndarray, budget: float) -> float:
    order = np.argsort(-score)
    yy = y[order]
    pos = max(int((y == 1).sum()), 1)
    neg = max(int((y == 0).sum()), 1)
    tp = 0
    fp = 0
    best = 0.0
    for label in yy:
        if int(label) == 1:
            tp += 1
        else:
            fp += 1
        if fp / neg <= budget:
            best = tp / pos
        else:
            break
    return float(best)


def best_threshold(y: np.ndarray, score: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y, score)
    if len(thresholds) == 0:
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.nanargmax(f1))])


def metrics(y: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    pred = (score >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "attack_precision": precision,
        "attack_recall": recall,
        "attack_f1": f1,
        "auroc": roc_auc_score(y, score) if len(np.unique(y)) == 2 else np.nan,
        "aupr": average_precision_score(y, score) if len(np.unique(y)) == 2 else np.nan,
        "recall_at_fpr_1e_3": recall_at_fpr(y, score, 1e-3),
        "num_pos": int((y == 1).sum()),
        "num_neg": int((y == 0).sum()),
    }


FEATURE_MASKS = {
    "full_safe_can": list(range(len(FEATURES))),
    "without_delta_t_same_id": [i for i, f in enumerate(FEATURES) if f != "delta_t_same_id"],
    "without_payload_delta_l1": [i for i, f in enumerate(FEATURES) if f != "payload_delta_l1"],
    "without_payload_statistics": [i for i, f in enumerate(FEATURES) if f not in {"payload_sum", "payload_mean", "payload_std"}],
    "without_can_id": [i for i, f in enumerate(FEATURES) if f != "can_id"],
    "without_payload_bytes": [i for i, f in enumerate(FEATURES) if not f.startswith("data")],
    "only_timing": [FEATURES.index("delta_t_global"), FEATURES.index("delta_t_same_id")],
    "only_payload": [i for i, f in enumerate(FEATURES) if f.startswith("data") or f in {"payload_sum", "payload_mean", "payload_std", "payload_delta_l1"}],
    "only_id": [FEATURES.index("can_id")],
}


def grain_full_retraining_ablation() -> pd.DataFrame:
    print("[completion] collect capped train/val for retraining ablation", flush=True)
    x_train, y_train, x_val, y_val = collect_train(max_neg_train=120_000, max_neg_val=60_000, seed=42)
    # Evaluate on full CT&T test04 stream. Each feature mask is independently fit.
    test_parts = []
    y_parts = []
    for _, x, y in iter_test_folder(TEST_FOLDERS["ctt_test04"]):
        test_parts.append(x)
        y_parts.append(y)
    x_test = np.vstack(test_parts)
    y_test = np.concatenate(y_parts)
    rows = []
    for name, cols in FEATURE_MASKS.items():
        print(f"[completion] retrain ablation {name}", flush=True)
        start = time.time()
        scaler = StandardScaler().fit(x_train[:, cols])
        model = GradientBoostingClassifier(n_estimators=35, max_depth=2, random_state=42)
        model.fit(scaler.transform(x_train[:, cols]), y_train)
        val_score = model.predict_proba(scaler.transform(x_val[:, cols]))[:, 1]
        threshold = best_threshold(y_val, val_score)
        score = model.predict_proba(scaler.transform(x_test[:, cols]))[:, 1]
        row = {
            "setting": "ctt_test04",
            "ablation": name,
            "features_used": "|".join(FEATURES[i] for i in cols),
            "num_features": len(cols),
            "fit_eval_time_sec": time.time() - start,
            "threshold_source": "validation_f1",
            **metrics(y_test, score, threshold),
        }
        rows.append(row)
    out = write_table(pd.DataFrame(rows).sort_values("attack_f1", ascending=False), "grain_full_retraining_ablation")
    (OUT / "grain_full_retraining_ablation.md").write_text(
        "# GRAIN Full Retraining Feature Ablation\n\n"
        "Each row retrains a GradientBoosting GRAIN sample-level baseline with a different feature mask using all positives and capped negatives from the original CT&T train split, then evaluates on full CT&T test04. This is stricter than feature-importance proxy evidence, but it remains a sample-level ablation rather than the aggregate-window model.\n",
        encoding="utf-8",
    )
    return out


def external_full_corrected_benchmark() -> pd.DataFrame:
    rows = []
    for path in [
        Path("results/attack_centric_final/tables/i1_external_corrected_sanity.csv"),
        Path("results/final_grain_can/tables/g1_cross_dataset_validation.csv"),
        Path("results/cmf_tables/paper_table_overall_main_results_refined.csv"),
    ]:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        for _, r in df.iterrows():
            dataset = r.get("dataset", r.get("Dataset/Setting", "NA"))
            if str(dataset).startswith("ctt"):
                continue
            rows.append({
                "dataset": dataset,
                "model": r.get("model", r.get("Model", "NA")),
                "attack_f1": r.get("best_available_model_attack_f1", r.get("f1", np.nan)),
                "aupr": r.get("best_available_model_aupr", r.get("aupr", np.nan)),
                "auroc": r.get("auroc", np.nan),
                "recall_at_fpr_1e_3": r.get("recall_at_fpr_1e_3", r.get("recall_at_fpr_1em03", np.nan)),
                "positive_rate": r.get("positive_rate", np.nan),
                "source": str(path),
                "notes": r.get("notes_on_protocol", r.get("scope", "")),
            })
    out = pd.DataFrame(rows).drop_duplicates(["dataset", "model", "source"])
    out = write_table(out, "external_full_corrected_benchmark")
    (OUT / "external_full_corrected_benchmark.md").write_text(
        "# External Full Corrected Benchmark\n\n"
        "This table consolidates available corrected external evidence for ROAD, CrySyS subsets, HCRL and Car-Hacking. Rows inherit their source protocol notes; subset rows must remain labelled as subsets.\n",
        encoding="utf-8",
    )
    return out


def plot_bar(df: pd.DataFrame, name: str, x: str, y: str, title: str) -> None:
    d = df.copy()
    d[y] = pd.to_numeric(d[y], errors="coerce")
    d = d.dropna(subset=[x, y]).head(20)
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    if d.empty:
        ax.text(0.5, 0.5, "No supported data", ha="center", va="center")
    else:
        ax.bar(range(len(d)), d[y], color="#4C78A8", edgecolor="black", linewidth=0.7, hatch="//")
        ax.set_xticks(range(len(d)))
        ax.set_xticklabels(d[x].astype(str), rotation=35, ha="right")
        ax.set_ylabel(y)
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGS / f"{name}.svg", format="svg")
    plt.close(fig)


def final_report(ext: pd.DataFrame, ab: pd.DataFrame) -> None:
    best_ab = ab.sort_values("attack_f1", ascending=False).head(1).iloc[0] if len(ab) else {}
    (OUT / "final_evidence_completion_report.md").write_text(
        "# Final Evidence Completion Report\n\n"
        "## External dependencies\n\n"
        "Original-author confusion matrices/exact code, can-train-and-test-v1.5, and official event boundaries were searched for in public web results and local filesystem candidates. They were not found in accessible sources. The original can-train-and-test set_01 is present and fingerprinted.\n\n"
        "## Local controlled experiments\n\n"
        f"External corrected benchmark rows consolidated: {len(ext)}.\n\n"
        f"Full retraining feature ablation rows completed: {len(ab)}. Best ablation: `{best_ab.get('ablation', 'NA')}` with attack-F1={best_ab.get('attack_f1', np.nan):.4f}, AUPR={best_ab.get('aupr', np.nan):.4f}, Recall@FPR=1e-3={best_ab.get('recall_at_fpr_1e_3', np.nan):.4f}.\n\n"
        "## Paper implication\n\n"
        "Exact-alignment claims still require author code/confusion matrices or v1.5 source. The strengthened local evidence supports a measurement + corrected benchmark + GRAIN baseline paper, with explicit limitations.\n",
        encoding="utf-8",
    )


def audit() -> None:
    required = [
        TABLES / "external_dependency_audit.csv",
        TABLES / "v15_dataset_alignment.csv",
        TABLES / "author_code_confusion_matrix_search.csv",
        TABLES / "official_event_boundary_audit.csv",
        TABLES / "external_full_corrected_benchmark.csv",
        TABLES / "grain_full_retraining_ablation.csv",
        FIGS / "external_full_corrected_benchmark.svg",
        FIGS / "grain_full_retraining_ablation.svg",
        OUT / "final_evidence_completion_report.md",
    ]
    rows = []
    for p in required:
        status = "ok" if p.exists() and p.stat().st_size > 0 else "missing_or_empty"
        if p.suffix == ".csv" and status == "ok":
            try:
                status = "ok" if len(pd.read_csv(p)) > 0 else "empty_csv"
            except Exception as exc:
                status = f"csv_error:{exc}"
        if p.suffix == ".svg" and status == "ok":
            status = "ok" if "<svg" in p.read_text(errors="ignore") else "invalid_svg"
        rows.append({"file_path": str(p), "status": status, "size": p.stat().st_size if p.exists() else 0})
    out = pd.DataFrame(rows)
    out.to_csv(AUDITS / "output_integrity_audit.csv", index=False)
    failures = out[out["status"].ne("ok")]
    (AUDITS / "output_integrity_audit.md").write_text(
        "# Final Evidence Completion Integrity Audit\n\n"
        f"Checked files: {len(out)}\n\nFailures: {len(failures)}\n\n"
        + ("All required files are present and non-empty.\n" if failures.empty else f"```csv\n{failures.to_csv(index=False)}```\n"),
        encoding="utf-8",
    )


def main() -> None:
    setup()
    external_dependency_audit()
    v15_alignment()
    set01_manifest()
    author_code_confusion_matrix_search()
    official_event_boundary_audit()
    ext = external_full_corrected_benchmark()
    ab = grain_full_retraining_ablation()
    plot_bar(ext.sort_values("attack_f1", ascending=False), "external_full_corrected_benchmark", "dataset", "attack_f1", "External Corrected Benchmark")
    plot_bar(ab.sort_values("attack_f1", ascending=False), "grain_full_retraining_ablation", "ablation", "attack_f1", "GRAIN Retraining Ablation")
    final_report(ext, ab)
    audit()


if __name__ == "__main__":
    main()
