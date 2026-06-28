from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from cmf_can.analysis.calibrate import collect_logits, evaluate_method, softmax_score
from cmf_can.data.collate import collate_batch
from cmf_can.data.dataset import CMFWindowDataset, SPLIT_TEST, SPLIT_VAL
from cmf_can.models.cmf import build_model


def _normalize(scores: np.ndarray) -> np.ndarray:
    scores = scores.astype(float)
    lo = float(np.min(scores))
    hi = float(np.max(scores))
    if hi - lo < 1e-12:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


def score_candidates(logits: np.ndarray) -> dict[str, np.ndarray]:
    probs = np.stack([1.0 - softmax_score(logits), softmax_score(logits)], axis=1)
    max_prob = probs.max(axis=1)
    entropy = -(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum(axis=1) / np.log(2.0)
    logsumexp = np.log(np.exp(logits - logits.max(axis=1, keepdims=True)).sum(axis=1)) + logits.max(axis=1)
    margin = logits[:, 1] - logits[:, 0]
    return {
        "attack_prob": probs[:, 1],
        "attack_margin": _normalize(margin),
        "uncertainty_entropy": entropy,
        "uncertainty_inverse_maxprob": 1.0 - max_prob,
        "energy_confidence": _normalize(logsumexp),
        "energy_ood": _normalize(-logsumexp),
    }


def run_scores(
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

    val_scores = score_candidates(val_logits)
    test_scores = score_candidates(test_logits)
    rows: list[dict] = []
    for score_name in val_scores:
        for policy in ["val_f1", "val_fpr_1em04", "val_fpr_5em04", "val_fpr_1em03"]:
            row = {
                "dataset": dataset,
                "model": model_name,
                "seed": seed,
                "label_ratio": label_ratio,
                "score": score_name,
                **evaluate_method(score_name, val_y, val_scores[score_name], test_y, test_scores[score_name], policy),
            }
            rows.append(row)

    out = root / "results/cmf_tables" / table
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    write_header = not out.exists()
    with out.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
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
    parser.add_argument("--table", default="ood_score_trials.csv")
    args = parser.parse_args()
    rows = run_scores(
        root=Path(args.root).resolve(),
        dataset=args.dataset,
        model_name=args.model,
        seed=args.seed,
        label_ratio=args.label_ratio,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        table=args.table,
    )
    best = max(rows, key=lambda row: row["f1"])
    print(
        f"[ood_scores] {args.dataset}/{args.model} seed={args.seed} "
        f"best={best['score']}/{best['threshold_policy']} "
        f"f1={best['f1']:.4f} aupr={best['aupr']:.4f} "
        f"recall@1e-4={best['recall_at_fpr_1em04']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
