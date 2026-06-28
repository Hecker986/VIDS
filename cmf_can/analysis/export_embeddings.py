from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Subset

from cmf_can.data.collate import collate_batch
from cmf_can.data.dataset import CMFWindowDataset, SPLIT_TEST
from cmf_can.models.cmf import build_model


def _to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def stratified_indices(ds: CMFWindowDataset, per_class: int, seed: int) -> np.ndarray:
    labels = ds.windows[:, 2].astype(int)
    rng = np.random.default_rng(seed)
    idx: list[int] = []
    for label in [0, 1]:
        pos = np.where(labels == label)[0]
        if len(pos):
            idx.extend(rng.choice(pos, size=min(per_class, len(pos)), replace=False).tolist())
    rng.shuffle(idx)
    return np.asarray(idx, dtype=np.int64)


@torch.no_grad()
def collect_embeddings(root: Path, dataset: str, model_name: str, per_class: int, seed: int, batch_size: int) -> tuple[list[dict], np.ndarray]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = CMFWindowDataset(root, dataset, SPLIT_TEST)
    idx = stratified_indices(ds, per_class, seed)
    loader = DataLoader(Subset(ds, idx.tolist()), batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_batch)
    model = build_model(model_name).to(device).eval()
    ckpt = root / "checkpoints/cmf" / dataset / model_name / f"ratio_1_seed_{seed}" / "best.pt"
    if not ckpt.exists():
        ckpt = root / "checkpoints/cmf" / dataset / model_name / "ratio_1_seed_42" / "best.pt"
    model.load_state_dict(torch.load(ckpt, map_location=device))
    rows: list[dict] = []
    embeddings: list[np.ndarray] = []
    for batch in loader:
        batch = _to_device(batch, device)
        logits = model(batch)
        emb = getattr(model, "last_embedding", None)
        if emb is None:
            continue
        score = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        e = emb.detach().cpu().numpy()
        embeddings.append(e)
        for i in range(e.shape[0]):
            rows.append(
                {
                    "sample_id": batch["sample_id"][i],
                    "dataset": dataset,
                    "model": model_name,
                    "label": int(batch["label"][i].detach().cpu().item()),
                    "score": float(score[i]),
                    "attack_type": batch["attack_type"][i],
                    "vehicle": batch["vehicle"][i],
                }
            )
    return rows, np.concatenate(embeddings, axis=0)


def save_outputs(root: Path, datasets: list[str], model: str, per_class: int, seed: int, batch_size: int) -> None:
    out_dir = root / "results/cmf_embeddings"
    fig_dir = root / "results/cmf_figures"
    table_dir = root / "results/cmf_tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    all_emb: list[np.ndarray] = []
    for dataset in datasets:
        rows, emb = collect_embeddings(root, dataset, model, per_class, seed, batch_size)
        emb_path = out_dir / f"{dataset}_{model}_embedding_sample.npy"
        csv_path = out_dir / f"{dataset}_{model}_embedding_sample.csv"
        np.save(emb_path, emb)
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) + [f"emb_{i}" for i in range(emb.shape[1])])
            writer.writeheader()
            for row, vec in zip(rows, emb):
                writer.writerow({**row, **{f"emb_{i}": float(v) for i, v in enumerate(vec)}})
        for row in rows:
            row["embedding_path"] = str(emb_path)
        all_rows.extend(rows)
        all_emb.append(emb)

    emb_all = np.concatenate(all_emb, axis=0)
    pca_dim = min(50, emb_all.shape[1], emb_all.shape[0] - 1)
    reduced = PCA(n_components=pca_dim, random_state=seed).fit_transform(emb_all)
    coords = TSNE(n_components=2, random_state=seed, init="pca", learning_rate="auto", perplexity=min(30, max(5, (len(all_rows) - 1) // 10))).fit_transform(reduced)
    manifest = table_dir / "paper_table_embedding_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "model", "sample_id", "label", "attack_type", "vehicle", "embedding_path", "tsne_x", "tsne_y"])
        writer.writeheader()
        for row, xy in zip(all_rows, coords):
            writer.writerow({**{k: row[k] for k in ["dataset", "model", "sample_id", "label", "attack_type", "vehicle", "embedding_path"]}, "tsne_x": float(xy[0]), "tsne_y": float(xy[1])})

    mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "svg.fonttype": "none", "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.3 * len(datasets), 3.0), sharex=False, sharey=False)
    axes = np.atleast_1d(axes)
    start = 0
    for ax, dataset, emb in zip(axes, datasets, all_emb):
        end = start + len(emb)
        part_rows = all_rows[start:end]
        labels = np.asarray([r["label"] for r in part_rows])
        ax.scatter(coords[start:end, 0], coords[start:end, 1], c=np.where(labels == 1, "#E45756", "#4C78A8"), s=8, alpha=0.72, linewidths=0)
        ax.set_title(dataset.replace("ctt_", "CT&T "))
        ax.set_xticks([])
        ax.set_yticks([])
        start = end
    handles = [
        mpl.lines.Line2D([0], [0], marker="o", color="w", label="normal", markerfacecolor="#4C78A8", markersize=5),
        mpl.lines.Line2D([0], [0], marker="o", color="w", label="attack", markerfacecolor="#E45756", markersize=5),
    ]
    axes[-1].legend(handles=handles, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=2)
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(fig_dir / f"paper_fig_tsne_embeddings.{ext}", dpi=300 if ext == "png" else None, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--datasets", nargs="+", default=["road", "ctt_test02", "ctt_test04"])
    parser.add_argument("--model", default="cmf_can")
    parser.add_argument("--per-class", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    save_outputs(args.root.resolve(), args.datasets, args.model, args.per_class, args.seed, args.batch_size)
    print("[write] embedding samples and t-SNE figure")


if __name__ == "__main__":
    main()
