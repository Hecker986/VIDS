"""Training loop for CMF-CAN experiments."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from cmf_can.data.collate import collate_batch
from cmf_can.data.dataset import CMFWindowDataset, SPLIT_TEST, SPLIT_TRAIN, SPLIT_VAL, stratified_subset
from cmf_can.models.cmf import build_model
from cmf_can.utils.metrics import best_threshold, compute_metrics, constrained_fpr_metrics
from cmf_can.utils.seed import set_seed


def _to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def _model_logits(model: torch.nn.Module, batch: dict, return_aux: bool = False):
    if return_aux:
        try:
            out = model(batch, return_aux=True)
        except TypeError:
            out = model(batch)
        if isinstance(out, tuple):
            return out
        return out, {}
    return model(batch)


def _class_weights(dataset, device: torch.device) -> torch.Tensor | None:
    if hasattr(dataset, "windows"):
        labels = dataset.windows[:, 2]
    elif hasattr(dataset, "dataset") and hasattr(dataset.dataset, "windows"):
        labels = dataset.dataset.windows[dataset.indices, 2]
    else:
        return None
    n0 = int((labels == 0).sum())
    n1 = int((labels == 1).sum())
    if n0 == 0 or n1 == 0:
        return None
    return torch.tensor([len(labels) / (2 * n0), len(labels) / (2 * n1)], dtype=torch.float32, device=device)


def _labels(dataset) -> np.ndarray | None:
    if hasattr(dataset, "windows"):
        return dataset.windows[:, 2].astype(np.int64)
    if hasattr(dataset, "dataset") and hasattr(dataset.dataset, "windows"):
        return dataset.dataset.windows[dataset.indices, 2].astype(np.int64)
    return None


def _weighted_sampler(dataset, seed: int) -> WeightedRandomSampler | None:
    labels = _labels(dataset)
    if labels is None:
        return None
    n0 = int((labels == 0).sum())
    n1 = int((labels == 1).sum())
    if n0 == 0 or n1 == 0:
        return None
    class_weight = {0: len(labels) / (2 * n0), 1: len(labels) / (2 * n1)}
    weights = torch.as_tensor([class_weight[int(y)] for y in labels], dtype=torch.double)
    gen = torch.Generator()
    gen.manual_seed(seed)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True, generator=gen)


class FocalLoss(torch.nn.Module):
    def __init__(self, weight: torch.Tensor | None = None, gamma: float = 2.0, alpha: float | None = None):
        super().__init__()
        self.register_buffer("weight", weight if weight is not None else None)
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = torch.nn.functional.cross_entropy(logits, target, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1.0 - pt) ** self.gamma) * ce
        if self.alpha is not None:
            alpha_t = torch.where(target == 1, self.alpha, 1.0 - self.alpha)
            loss = alpha_t * loss
        return loss.mean()


def supervised_contrastive_loss(embedding: torch.Tensor, labels: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    labels = labels.view(-1, 1)
    positive = torch.eq(labels, labels.T).float().to(embedding.device)
    positive.fill_diagonal_(0.0)
    if positive.sum() == 0:
        return embedding.new_zeros(())
    z = torch.nn.functional.normalize(embedding.float(), dim=1)
    logits = z @ z.T / max(float(temperature), 1e-6)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    logits_mask = torch.ones_like(positive)
    logits_mask.fill_diagonal_(0.0)
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    pos_count = positive.sum(dim=1)
    valid = pos_count > 0
    mean_log_prob_pos = (positive * log_prob).sum(dim=1)[valid] / pos_count[valid]
    return -mean_log_prob_pos.mean()


def _selection_score(metric_name: str, y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> tuple[float, float]:
    if metric_name in {"f1", "macro_f1"}:
        metrics = compute_metrics(y_true, y_score, threshold)
        return float(metrics[metric_name]), threshold
    if metric_name == "aupr":
        metrics = compute_metrics(y_true, y_score, threshold)
        return float(metrics["aupr"]), threshold
    if metric_name in {"recall_at_fpr_1em04", "f1_at_fpr_1em04", "recall_at_fpr_5em04", "f1_at_fpr_5em04", "recall_at_fpr_1em03", "f1_at_fpr_1em03"}:
        metrics = constrained_fpr_metrics(y_true, y_score)
        suffix = metric_name.split("_at_fpr_", 1)[1]
        return float(metrics[metric_name]), float(metrics[f"threshold_at_fpr_{suffix}"])
    raise ValueError(f"unknown selection metric: {metric_name}")


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    labels: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    for batch in loader:
        batch = _to_device(batch, device)
        logits = _model_logits(model, batch)
        prob = torch.softmax(logits, dim=1)[:, 1]
        labels.append(batch["label"].detach().cpu().numpy())
        scores.append(prob.detach().cpu().numpy())
    return np.concatenate(labels), np.concatenate(scores)


@torch.no_grad()
def prediction_dump(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    dataset: str,
    model_name: str,
    threshold: float,
    save_gate_weights: bool,
) -> tuple[list[dict], list[dict]]:
    model.eval()
    pred_rows: list[dict] = []
    gate_rows: list[dict] = []
    setting = dataset
    for batch in loader:
        batch = _to_device(batch, device)
        if save_gate_weights:
            logits, aux = _model_logits(model, batch, return_aux=True)
        else:
            logits = _model_logits(model, batch)
            aux = {}
        prob = torch.softmax(logits, dim=1)[:, 1]
        labels = batch["label"].detach().cpu().numpy().astype(int)
        scores = prob.detach().cpu().numpy()
        preds = (scores >= threshold).astype(int)
        n = len(labels)
        meta = {
            "sample_id": batch.get("sample_id", ["NA"] * n),
            "attack_type": batch.get("attack_type", ["NA"] * n),
            "vehicle": batch.get("vehicle", ["NA"] * n),
            "window_start": batch.get("window_start", ["NA"] * n),
            "window_end": batch.get("window_end", ["NA"] * n),
            "split": batch.get("split", ["test"] * n),
        }
        aux_arrays = {
            k: v.detach().cpu().numpy()
            for k, v in aux.items()
            if torch.is_tensor(v) and v.ndim == 1 and (
                k.startswith("gate_")
                or k.startswith("reliability_")
                or k in {"context_shift_score", "context_mask_value", "topk_score"}
            )
        }
        for i in range(n):
            row = {
                "sample_id": meta["sample_id"][i],
                "dataset": dataset,
                "setting": setting,
                "model": model_name,
                "label": int(labels[i]),
                "prediction": int(preds[i]),
                "score": float(scores[i]),
                "threshold": float(threshold),
                "attack_type": meta["attack_type"][i],
                "vehicle": meta["vehicle"][i],
                "window_start": meta["window_start"][i],
                "window_end": meta["window_end"][i],
                "split": meta["split"][i],
            }
            pred_rows.append(row)
            if {"gate_frame", "gate_window", "gate_context"}.issubset(aux_arrays):
                aux_row = {}
                for key in [
                    "gate_frame",
                    "gate_window",
                    "gate_context",
                    "reliability_frame",
                    "reliability_window",
                    "reliability_context",
                    "context_shift_score",
                    "context_mask_value",
                    "topk_score",
                ]:
                    if key in aux_arrays:
                        aux_row[key] = float(aux_arrays[key][i])
                gate_rows.append(
                    {
                        **{k: row[k] for k in ["sample_id", "dataset", "setting", "model", "label", "prediction", "score", "attack_type", "vehicle"]},
                        **aux_row,
                    }
                )
    return pred_rows, gate_rows


def write_prediction_outputs(root: Path, dataset: str, model_name: str, pred_rows: list[dict], gate_rows: list[dict]) -> None:
    out_dir = root / "results/cmf_predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / f"{dataset}_{model_name}_predictions.csv"
    with pred_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "dataset",
                "setting",
                "model",
                "label",
                "prediction",
                "score",
                "threshold",
                "attack_type",
                "vehicle",
                "window_start",
                "window_end",
                "split",
            ],
        )
        writer.writeheader()
        writer.writerows(pred_rows)
    if gate_rows:
        gate_path = out_dir / f"{dataset}_{model_name}_gate_weights.csv"
        with gate_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "sample_id",
                    "dataset",
                    "setting",
                    "model",
                    "label",
                    "prediction",
                    "score",
                    "attack_type",
                    "vehicle",
                    "gate_frame",
                    "gate_window",
                    "gate_context",
                    "reliability_frame",
                    "reliability_window",
                    "reliability_context",
                    "context_shift_score",
                    "context_mask_value",
                    "topk_score",
                ],
            )
            writer.writeheader()
            writer.writerows(gate_rows)


@torch.no_grad()
def write_embedding_outputs(root: Path, dataset: str, model_name: str, model: torch.nn.Module, loader: DataLoader, device: torch.device) -> None:
    model.eval()
    out_dir = root / "results/cmf_predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    embeddings: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    sample_ids: list[str] = []
    attack_types: list[str] = []
    for batch in loader:
        batch = _to_device(batch, device)
        _ = _model_logits(model, batch)
        emb = getattr(model, "last_embedding", None)
        if emb is None:
            continue
        n = int(emb.shape[0])
        embeddings.append(emb.detach().cpu().numpy())
        labels.append(batch["label"].detach().cpu().numpy().astype(int))
        sample_ids.extend([str(x) for x in batch.get("sample_id", ["NA"] * n)])
        attack_types.extend([str(x) for x in batch.get("attack_type", ["NA"] * n)])
    if not embeddings:
        return
    np.savez_compressed(
        out_dir / f"{dataset}_{model_name}_embeddings.npz",
        embedding=np.concatenate(embeddings, axis=0),
        label=np.concatenate(labels, axis=0),
        attack_type=np.asarray(attack_types, dtype=object),
        sample_id=np.asarray(sample_ids, dtype=object),
        setting=np.asarray([dataset] * len(sample_ids), dtype=object),
        model=np.asarray([model_name] * len(sample_ids), dtype=object),
    )


def run_training(
    root: Path,
    dataset: str,
    model_name: str,
    epochs: int = 20,
    batch_size: int = 512,
    lr: float = 5e-5,
    seed: int = 42,
    label_ratio: float = 1.0,
    table: str | None = None,
    num_workers: int = 2,
    aux_loss_weight: float = 0.2,
    loss_name: str = "ce",
    focal_gamma: float = 2.0,
    focal_alpha: float | None = None,
    sampler_name: str = "none",
    selection_metric: str = "f1",
    class_weights_name: str = "balanced",
    supcon_weight: float = 0.0,
    supcon_temperature: float = 0.1,
    eval_only: bool = False,
    checkpoint: str | None = None,
    save_predictions: bool = False,
    save_gate_weights: bool = False,
    save_embeddings: bool = False,
) -> dict:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_base = CMFWindowDataset(root, dataset, SPLIT_TRAIN)
    val_ds = CMFWindowDataset(root, dataset, SPLIT_VAL)
    test_ds = CMFWindowDataset(root, dataset, SPLIT_TEST)
    train_ds = stratified_subset(train_base, label_ratio, seed)

    model = build_model(model_name).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    if class_weights_name == "balanced":
        class_weights = _class_weights(train_ds, device)
    elif class_weights_name == "none":
        class_weights = None
    else:
        raise ValueError(f"unknown class weights: {class_weights_name}")
    if loss_name == "ce":
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    elif loss_name == "focal":
        loss_fn = FocalLoss(weight=class_weights, gamma=focal_gamma, alpha=focal_alpha)
    else:
        raise ValueError(f"unknown loss: {loss_name}")
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    sampler = _weighted_sampler(train_ds, seed) if sampler_name == "weighted" else None
    if sampler_name not in {"none", "weighted"}:
        raise ValueError(f"unknown sampler: {sampler_name}")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=sampler is None, sampler=sampler, collate_fn=collate_batch, num_workers=num_workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch, num_workers=num_workers, pin_memory=device.type == "cuda")
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch, num_workers=num_workers, pin_memory=device.type == "cuda")

    ckpt_dir = root / "checkpoints/cmf" / dataset / model_name / f"ratio_{label_ratio:g}_seed_{seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_state = None
    best_f1 = -1.0
    best_t = 0.5
    if eval_only:
        ckpt = Path(checkpoint) if checkpoint else ckpt_dir / "best.pt"
        if not ckpt.is_absolute():
            ckpt = root / ckpt
        model.load_state_dict(torch.load(ckpt, map_location=device))
        y_val, s_val = evaluate(model, val_loader, device)
        best_t = best_threshold(y_val, s_val, metric="f1")
        best_f1, best_t = _selection_score(selection_metric, y_val, s_val, best_t)
    else:
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
            for batch in train_loader:
                batch = _to_device(batch, device)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    logits = _model_logits(model, batch)
                    loss = loss_fn(logits, batch["label"])
                    aux_logits = getattr(model, "last_aux_logits", None)
                    if aux_logits is not None and aux_loss_weight > 0:
                        aux_indices = getattr(model, "active_aux_indices", range(len(aux_logits)))
                        active_aux = [aux_logits[i] for i in aux_indices]
                        loss = loss + aux_loss_weight * sum(loss_fn(aux, batch["label"]) for aux in active_aux) / len(active_aux)
                    embedding = getattr(model, "last_embedding", None)
                    if embedding is not None and supcon_weight > 0:
                        loss = loss + supcon_weight * supervised_contrastive_loss(embedding, batch["label"], supcon_temperature)
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(opt)
                scaler.update()
                total_loss += float(loss.item())
            scheduler.step()
            y_val, s_val = evaluate(model, val_loader, device)
            threshold = best_threshold(y_val, s_val, metric="f1")
            val_metrics = compute_metrics(y_val, s_val, threshold)
            select_value, select_threshold = _selection_score(selection_metric, y_val, s_val, threshold)
            print(
                f"[CMF {dataset}/{model_name}] ep {epoch}/{epochs} "
                f"loss={total_loss / max(len(train_loader), 1):.4f} "
                f"val_f1={val_metrics['f1']:.4f} val_aupr={val_metrics['aupr']:.4f} "
                f"select_{selection_metric}={select_value:.4f} thr={select_threshold:.4f}",
                flush=True,
            )
            if select_value > best_f1:
                best_f1 = select_value
                best_t = select_threshold
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None and not eval_only:
        model.load_state_dict(best_state)
        torch.save(best_state, ckpt_dir / "best.pt")
    if not eval_only:
        torch.save(model.state_dict(), ckpt_dir / "last.pt")

    y_test, s_test = evaluate(model, test_loader, device)
    result = {
        "dataset": dataset,
        "model": model_name,
        "seed": seed,
        "label_ratio": label_ratio,
        "epochs": epochs,
        "aux_loss_weight": aux_loss_weight,
        "loss": loss_name,
        "focal_gamma": focal_gamma,
        "focal_alpha": focal_alpha if focal_alpha is not None else "",
        "sampler": sampler_name,
        "selection_metric": selection_metric,
        "class_weights": class_weights_name,
        "supcon_weight": supcon_weight,
        "supcon_temperature": supcon_temperature,
        **compute_metrics(y_test, s_test, best_t),
        **constrained_fpr_metrics(y_test, s_test),
    }
    print(f"[CMF {dataset}/{model_name}] test {result}", flush=True)
    if save_predictions:
        pred_rows, gate_rows = prediction_dump(model, test_loader, device, dataset, model_name, best_t, save_gate_weights)
        write_prediction_outputs(root, dataset, model_name, pred_rows, gate_rows)
    if save_embeddings:
        write_embedding_outputs(root, dataset, model_name, model, test_loader, device)
    if table:
        out = root / "results/cmf_tables" / table
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            with out.open("r", newline="", encoding="utf-8") as f:
                existing_header = next(csv.reader(f), None)
            if existing_header and existing_header != list(result.keys()):
                out = out.with_name(f"{out.stem}_v2{out.suffix}")
        write_header = not out.exists()
        with out.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(result.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(result)
    return result
