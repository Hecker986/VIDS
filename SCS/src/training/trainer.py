"""Unified training and evaluation loop."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from src.datasets.collate import collate_batch
from src.datasets.dynamic_window_dataset import DynamicWindowDataset, load_dataset_bundle
from src.datasets.ssl_window_dataset import SSLWindowDataset
from src.models.baselines.cnn import CANCNN
from src.models.baselines.lstm import CANLSTM
from src.models.baselines.transformer import CANTransformer
from src.models.scs_can import SCSCAN, build_model
from src.utils.metrics import compute_metrics, find_best_threshold
from src.utils.schema import SPLIT_TEST, SPLIT_TRAIN, SPLIT_VAL
from src.utils.seed import set_seed


def build_arch(model_name: str, variant: str = "full"):
    if model_name == "cnn":
        return CANCNN()
    if model_name == "lstm":
        return CANLSTM()
    if model_name == "transformer":
        return CANTransformer()
    if model_name.startswith("scs_can"):
        return build_model(variant)
    raise ValueError(model_name)


def _compute_class_weights(dataset, device) -> torch.Tensor:
    if hasattr(dataset, "windows"):
        labels = dataset.windows[:, 2]
    elif hasattr(dataset, "dataset") and hasattr(dataset.dataset, "windows"):
        labels = dataset.dataset.windows[:, 2]
    else:
        return None
    n0 = int((labels == 0).sum())
    n1 = int((labels == 1).sum())
    if n1 == 0 or n0 == 0:
        return None
    w0 = len(labels) / (2.0 * n0)
    w1 = len(labels) / (2.0 * n1)
    return torch.tensor([w0, w1], dtype=torch.float32, device=device)


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[dict, np.ndarray, np.ndarray]:
    model.eval()
    scores, labels = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch)
        prob = torch.softmax(logits, dim=1)[:, 1]
        scores.extend(prob.cpu().numpy())
        labels.extend(batch["label"].cpu().numpy())
    y = np.asarray(labels)
    s = np.asarray(scores)
    return compute_metrics(y, s), y, s


def finetune(root: Path, dataset: str, model_name: str, variant: str = "full",
             epochs: int = 20, batch_size: int = 512, lr: float = 5e-5,
             seed: int = 42, label_ratio: float = 1.0, pretrained: Path | None = None,
             ipc_weight: float = 0.0) -> dict:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = load_dataset_bundle(root, dataset, SPLIT_TRAIN)
    val_ds = load_dataset_bundle(root, dataset, SPLIT_VAL)
    test_ds = load_dataset_bundle(root, dataset, SPLIT_TEST)
    if label_ratio < 1.0:
        n = max(1, int(len(train_ds) * label_ratio))
        idx = np.random.choice(len(train_ds), n, replace=False)
        train_ds = Subset(train_ds, idx.tolist())
    model = build_arch(model_name, variant).to(device)
    if pretrained and Path(pretrained).exists() and isinstance(model, SCSCAN):
        model.load_state_dict(torch.load(pretrained, map_location=device, weights_only=True))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    class_w = _compute_class_weights(train_ds, device)
    ce = torch.nn.CrossEntropyLoss(weight=class_w)
    bce = torch.nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == "cuda")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_batch, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch, num_workers=0)
    ckpt_dir = root / "checkpoints" / dataset / f"{model_name}_{variant}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_f1, best_state = -1.0, None
    for ep in range(epochs):
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=device.type == "cuda"):
                logits = model(batch)
                loss = ce(logits, batch["label"])
                if ipc_weight > 0 and isinstance(model, SCSCAN) and model.use_ipc:
                    ident = model.id_emb(batch["can_id"])
                    pay = model.payload_enc(batch["payload"])
                    ipc = model.ipc(torch.cat([ident, pay], -1)).squeeze(-1).mean(1)
                    match = torch.ones_like(ipc)
                    loss = loss + ipc_weight * bce(ipc, match)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        val_m, val_y, val_s = evaluate(model, val_loader, device)
        best_t = find_best_threshold(val_y, val_s)
        val_m_tuned = compute_metrics(val_y, val_s, threshold=best_t)
        f1 = val_m_tuned.get("f1", 0.0) or 0.0
        print(f"[{dataset}] ep {ep+1}/{epochs} val_auc={val_m['auroc']:.4f} val_f1={f1:.4f} thr={best_t:.3f}", flush=True)
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_threshold = best_t
    if best_state:
        model.load_state_dict(best_state)
        torch.save(best_state, ckpt_dir / "best.pt")
    else:
        best_threshold = 0.5
    torch.save(model.state_dict(), ckpt_dir / "last.pt")
    test_m, y, s = evaluate(model, test_loader, device)
    test_m_tuned = compute_metrics(y, s, threshold=best_threshold)
    print(f"[{dataset}] test with threshold={best_threshold:.3f}", flush=True)
    result = {"dataset": dataset, "model": model_name, "variant": variant, "seed": seed,
              "label_ratio": label_ratio, **test_m_tuned}
    print(result, flush=True)
    return result


def pretrain(root: Path, dataset: str, epochs: int = 10, batch_size: int = 512,
             lr: float = 1e-4, seed: int = 42) -> Path:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = load_dataset_bundle(root, dataset, SPLIT_TRAIN)
    ds = SSLWindowDataset(base)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate_batch, num_workers=4)
    model = build_model("full").to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    ce = torch.nn.CrossEntropyLoss(ignore_index=-100)
    bce = torch.nn.BCEWithLogitsLoss()
    ckpt_dir = root / "checkpoints" / dataset / "pretrain"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    for ep in range(epochs):
        model.train()
        total = 0.0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model.pretrain_forward(batch)
            loss = torch.tensor(0.0, device=device)
            if "id_logits" in out:
                mask = batch["id_mask"]
                logits = out["id_logits"][mask]
                tgt = batch["target_id"][mask]
                if len(tgt):
                    loss = loss + ce(logits, tgt)
                bmask = batch["pay_mask"]
                blogits = out["byte_logits"][bmask]
                btgt = batch["target_pay"][bmask]
                if len(btgt):
                    loss = loss + ce(blogits, btgt)
            if "ipc_logit" in out:
                loss = loss + bce(out["ipc_logit"], batch["ipc_match"].float())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.item())
        print(f"[pretrain {dataset}] ep {ep+1}/{epochs} loss={total/len(loader):.4f}", flush=True)
    path = ckpt_dir / "best.pt"
    torch.save(model.state_dict(), path)
    torch.save(model.state_dict(), ckpt_dir / "last.pt")
    return path
