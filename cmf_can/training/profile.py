from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from cmf_can.data.collate import collate_batch
from cmf_can.data.dataset import CMFWindowDataset, SPLIT_TEST
from cmf_can.models.cmf import build_model
from cmf_can.training.train import _to_device


def count_params(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


@torch.no_grad()
def profile_model(
    root: Path,
    dataset: str,
    model_name: str,
    batch_size: int,
    warmup: int,
    steps: int,
    num_workers: int,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = CMFWindowDataset(root, dataset, SPLIT_TEST)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    model = build_model(model_name).to(device).eval()
    total_params, trainable_params = count_params(model)

    times: list[float] = []
    examples = 0
    it = iter(loader)
    for i in range(warmup + steps):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        batch = _to_device(batch, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        _ = model(batch)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        if i >= warmup:
            times.append(elapsed)
            examples += int(batch["label"].shape[0])

    total_time = sum(times)
    return {
        "dataset": dataset,
        "model": model_name,
        "device": device.type,
        "batch_size": batch_size,
        "warmup_steps": warmup,
        "profile_steps": steps,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "avg_batch_ms": (total_time / max(len(times), 1)) * 1000.0,
        "throughput_windows_per_s": examples / max(total_time, 1e-12),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--dataset", default="road")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--table", default="efficiency_road.csv")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    rows = [
        profile_model(
            root=root,
            dataset=args.dataset,
            model_name=model,
            batch_size=args.batch_size,
            warmup=args.warmup,
            steps=args.steps,
            num_workers=args.num_workers,
        )
        for model in args.models
    ]
    out = root / "results/cmf_tables" / args.table
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row, flush=True)


if __name__ == "__main__":
    main()
