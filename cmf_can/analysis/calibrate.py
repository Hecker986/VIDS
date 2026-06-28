from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader

from cmf_can.data.collate import collate_batch
from cmf_can.data.dataset import CMFWindowDataset, SPLIT_TEST, SPLIT_VAL
from cmf_can.models.cmf import build_model
from cmf_can.utils.metrics import best_threshold, compute_metrics, constrained_fpr_metrics


def _to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


@torch.no_grad()
def collect_logits(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    labels: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    for batch in loader:
        batch = _to_device(batch, device)
        out = model(batch)
        labels.append(batch["label"].detach().cpu().numpy())
        logits.append(out.detach().cpu().numpy())
    return np.concatenate(labels), np.concatenate(logits)


def softmax_score(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits / max(float(temperature), 1e-6)
    z = z - z.max(axis=1, keepdims=True)
    exp = np.exp(z)
    return exp[:, 1] / exp.sum(axis=1)


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    x = torch.as_tensor(logits, dtype=torch.float32)
    y = torch.as_tensor(labels, dtype=torch.long)
    log_t = torch.zeros((), requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=80)
    loss_fn = torch.nn.CrossEntropyLoss()

    def closure():
        opt.zero_grad()
        loss = loss_fn(x / torch.exp(log_t).clamp_min(1e-6), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.exp(log_t).detach().clamp(1e-3, 1e3))


def fit_platt(logits: np.ndarray, labels: np.ndarray) -> LogisticRegression:
    margin = (logits[:, 1] - logits[:, 0]).reshape(-1, 1)
    clf = LogisticRegression(solver="lbfgs", class_weight="balanced", max_iter=1000)
    clf.fit(margin, labels)
    return clf


def platt_score(clf: LogisticRegression, logits: np.ndarray) -> np.ndarray:
    margin = (logits[:, 1] - logits[:, 0]).reshape(-1, 1)
    return clf.predict_proba(margin)[:, 1]


def expected_calibration_error(labels: np.ndarray, scores: np.ndarray, bins: int = 15) -> float:
    labels = labels.astype(int)
    scores = scores.astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (scores >= lo) & (scores < hi if hi < 1.0 else scores <= hi)
        if not mask.any():
            continue
        conf = scores[mask].mean()
        acc = labels[mask].mean()
        ece += mask.mean() * abs(acc - conf)
    return float(ece)


def threshold_for_val_fpr(labels: np.ndarray, scores: np.ndarray, key: str) -> float:
    metrics = constrained_fpr_metrics(labels, scores)
    return float(metrics[f"threshold_at_fpr_{key}"])


def evaluate_method(
    method: str,
    val_y: np.ndarray,
    val_scores: np.ndarray,
    test_y: np.ndarray,
    test_scores: np.ndarray,
    threshold_policy: str,
) -> dict:
    if threshold_policy == "val_f1":
        threshold = best_threshold(val_y, val_scores, metric="f1")
    elif threshold_policy == "val_fpr_1em04":
        threshold = threshold_for_val_fpr(val_y, val_scores, "1em04")
    elif threshold_policy == "val_fpr_5em04":
        threshold = threshold_for_val_fpr(val_y, val_scores, "5em04")
    elif threshold_policy == "val_fpr_1em03":
        threshold = threshold_for_val_fpr(val_y, val_scores, "1em03")
    else:
        raise ValueError(threshold_policy)
    out = {
        "calibration": method,
        "threshold_policy": threshold_policy,
        "val_threshold": float(threshold),
        "val_ece": expected_calibration_error(val_y, val_scores),
        "test_ece": expected_calibration_error(test_y, test_scores),
        **compute_metrics(test_y, test_scores, threshold),
        **constrained_fpr_metrics(test_y, test_scores),
    }
    return out


def run_calibration(
    root: Path,
    dataset: str,
    model_name: str,
    seed: int,
    label_ratio: float,
    batch_size: int,
    num_workers: int,
    table: str,
) -> list[dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_name).to(device)
    ckpt = root / "checkpoints/cmf" / dataset / model_name / f"ratio_{label_ratio:g}_seed_{seed}" / "best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)
    model.load_state_dict(torch.load(ckpt, map_location=device))

    val_ds = CMFWindowDataset(root, dataset, SPLIT_VAL)
    test_ds = CMFWindowDataset(root, dataset, SPLIT_TEST)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch, num_workers=num_workers, pin_memory=device.type == "cuda")
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch, num_workers=num_workers, pin_memory=device.type == "cuda")
    val_y, val_logits = collect_logits(model, val_loader, device)
    test_y, test_logits = collect_logits(model, test_loader, device)

    raw_val = softmax_score(val_logits)
    raw_test = softmax_score(test_logits)

    temp = fit_temperature(val_logits, val_y)
    temp_val = softmax_score(val_logits, temp)
    temp_test = softmax_score(test_logits, temp)

    platt = fit_platt(val_logits, val_y)
    platt_val = platt_score(platt, val_logits)
    platt_test = platt_score(platt, test_logits)

    rows: list[dict] = []
    for method, val_scores, test_scores, extra in [
        ("raw", raw_val, raw_test, {}),
        ("temperature", temp_val, temp_test, {"temperature": temp}),
        ("platt", platt_val, platt_test, {"platt_coef": float(platt.coef_[0, 0]), "platt_intercept": float(platt.intercept_[0])}),
    ]:
        for policy in ["val_f1", "val_fpr_1em04", "val_fpr_5em04", "val_fpr_1em03"]:
            row = {
                "dataset": dataset,
                "model": model_name,
                "seed": seed,
                "label_ratio": label_ratio,
                **extra,
                **evaluate_method(method, val_y, val_scores, test_y, test_scores, policy),
            }
            rows.append(row)

    out = root / "results/cmf_tables" / table
    out.parent.mkdir(parents=True, exist_ok=True)
    all_fields = []
    for row in rows:
        for key in row:
            if key not in all_fields:
                all_fields.append(key)
    write_header = not out.exists()
    with out.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_fields, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label-ratio", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--table", default="calibration_trials.csv")
    args = parser.parse_args()
    rows = run_calibration(
        root=Path(args.root).resolve(),
        dataset=args.dataset,
        model_name=args.model,
        seed=args.seed,
        label_ratio=args.label_ratio,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        table=args.table,
    )
    for row in rows:
        print(
            f"[calibrate] {row['dataset']}/{row['model']} seed={row['seed']} "
            f"{row['calibration']} {row['threshold_policy']} "
            f"f1={row['f1']:.4f} aupr={row['aupr']:.4f} "
            f"recall@1e-4={row['recall_at_fpr_1em04']:.4f} ece={row['test_ece']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
