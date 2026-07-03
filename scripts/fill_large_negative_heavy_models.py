from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cmf_can.analysis.final_grain_can_revision import compute_metrics, recall_at_budget
from cmf_can.analysis.protocol_rescue import TEST_FOLDERS, collect_train, iter_test_folder


ROOT = Path("results/final_grain_can")
TABLE = ROOT / "tables/a1_official_sample_negative_stability.csv"
OUT_COMPLETION = Path("results/final_evidence_completion")
LOGS = OUT_COMPLETION / "logs"
TABLES = OUT_COMPLETION / "tables"


def best_threshold(y: np.ndarray, score: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y, score)
    if len(thresholds) == 0:
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.nanargmax(f1))])


def model_bank(seed: int):
    return {
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=24,
            max_depth=16,
            min_samples_leaf=4,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=24,
            max_depth=16,
            min_samples_leaf=4,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(48,),
            max_iter=18,
            early_stopping=True,
            n_iter_no_change=4,
            random_state=seed,
        ),
    }


def eval_streaming(scaler, model, threshold: float, setting: str):
    y_parts, s_parts = [], []
    start = time.time()
    for _, x, y in iter_test_folder(TEST_FOLDERS[setting]):
        y_parts.append(y)
        s_parts.append(model.predict_proba(scaler.transform(x))[:, 1])
    y = np.concatenate(y_parts)
    score = np.concatenate(s_parts)
    metrics = compute_metrics(y, score, threshold)
    metrics["recall_at_fpr_1e_3"] = recall_at_budget(y, score, 1e-3)["recall"]
    metrics["auroc"] = roc_auc_score(y, score) if len(np.unique(y)) == 2 else np.nan
    metrics["aupr"] = average_precision_score(y, score) if len(np.unique(y)) == 2 else np.nan
    return y, metrics, time.time() - start


def main() -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    if not TABLE.exists():
        raise FileNotFoundError(TABLE)

    df = pd.read_csv(TABLE)
    comp_path = TABLES / "large_negative_heavy_model_completion.csv"
    if comp_path.exists():
        comp = pd.read_csv(comp_path)
    else:
        todo = df[df["status"].astype(str).eq("not_completed_resource_limit")].copy()
        todo = todo[todo["model"].isin(["ExtraTrees", "RandomForest", "MLP"])]
        todo = todo[todo["protocol"].isin(["B_2x_negative_cap", "C_5x_negative_cap"])]
        scenarios = (
            todo[["protocol", "negative_cap", "seed"]]
            .drop_duplicates()
            .sort_values(["protocol", "seed"])
            .to_dict("records")
        )
        completed = []

        for scenario in scenarios:
            protocol = str(scenario["protocol"])
            cap = int(float(scenario["negative_cap"]))
            seed = int(float(scenario["seed"]))
            print(f"[heavy-fill] protocol={protocol} cap={cap} seed={seed}", flush=True)
            x_train, y_train, x_val, y_val = collect_train(max_neg_train=cap, max_neg_val=max(30_000, cap // 2), seed=seed)
            scaler = StandardScaler().fit(x_train)
            xs = scaler.transform(x_train)
            xv = scaler.transform(x_val)
            for name, model in model_bank(seed).items():
                print(f"[heavy-fill] fit {name}", flush=True)
                start = time.time()
                model.fit(xs, y_train)
                fit_time = time.time() - start
                val_score = model.predict_proba(xv)[:, 1]
                threshold = best_threshold(y_val, val_score)
                for setting in TEST_FOLDERS:
                    print(f"[heavy-fill] eval {name} {setting}", flush=True)
                    y, metrics, inference_time = eval_streaming(scaler, model, threshold, setting)
                    completed.append(
                        {
                            "protocol": protocol,
                            "negative_cap": cap,
                            "negative_ratio": (len(y_train) - int(y_train.sum())) / max(int(y_train.sum()), 1),
                            "seed": seed,
                            "model": name,
                            "granularity": "sample",
                            "window_size": 1,
                            "setting": setting,
                            **metrics,
                            "num_train_pos": int(y_train.sum()),
                            "num_train_neg": int((y_train == 0).sum()),
                            "num_test_pos": int((y == 1).sum()),
                            "num_test_neg": int((y == 0).sum()),
                            "fit_time": fit_time,
                            "inference_time": inference_time,
                            "status": "completed_lightweight_large_negative",
                            "notes": "large-negative lightweight completion: capped negatives with constrained tree/MLP settings for final evidence closure",
                        }
                    )

        comp = pd.DataFrame(completed)
        comp.to_csv(comp_path, index=False)
        (TABLES / "large_negative_heavy_model_completion.tex").write_text(
            comp.to_latex(index=False, escape=True, na_rep="NA", float_format=lambda x: f"{x:.4f}"),
            encoding="utf-8",
        )

    if not comp.empty:
        key_cols = ["protocol", "seed", "model", "setting"]
        df2 = df.copy()
        for col in comp.columns:
            if col not in df2.columns:
                df2[col] = np.nan
        for _, row in comp.iterrows():
            mask = np.ones(len(df2), dtype=bool)
            for col in key_cols:
                mask &= df2[col].astype(str).eq(str(row[col]))
            if mask.any():
                idx = df2.index[mask]
                for col in comp.columns:
                    df2.loc[idx, col] = row[col]
            else:
                df2 = pd.concat([df2, row.to_frame().T], ignore_index=True)
        df2.to_csv(TABLE, index=False)
        (ROOT / "tables/a1_official_sample_negative_stability.tex").write_text(
            df2.to_latex(index=False, escape=True, na_rep="NA", float_format=lambda x: f"{x:.4f}"),
            encoding="utf-8",
        )

    (OUT_COMPLETION / "large_negative_heavy_model_completion.md").write_text(
        "# Large-Negative Heavy Model Completion\n\n"
        f"Completed rows: {len(comp)}.\n\n"
        "Rows replace earlier resource-limit placeholders for ExtraTrees, RandomForest, and MLP under Protocol B/C with constrained but real large-negative training/evaluation runs. "
        "Exact exhaustive full-negative Protocol D remains separately marked as protocol evidence rather than claimed as completed.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
