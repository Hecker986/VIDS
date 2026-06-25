"""SCS-CAN model and ablation variants."""
from __future__ import annotations

import torch
from torch import nn

from src.models.encoders import PayloadEncoder, TimeEncoder, TransitionContextEncoder


class SCSCAN(nn.Module):
    def __init__(self, vocab_size=4098, d_id=128, d_model=256, d_state=64,
                 num_layers=4, num_classes=2, use_time=True, use_transition=True,
                 use_mfm=True, use_ipc=True):
        super().__init__()
        self.use_time = use_time
        self.use_transition = use_transition
        self.use_mfm = use_mfm
        self.use_ipc = use_ipc
        self.id_emb = nn.Embedding(vocab_size, d_id)
        self.payload_enc = PayloadEncoder(d_payload=128)
        self.time_enc = TimeEncoder(d_time=32) if use_time else None
        self.trans_enc = TransitionContextEncoder(self.id_emb, d_state) if use_transition else None
        in_dim = d_id + 128 + (32 if use_time else 0) + (d_state if use_transition else 0)
        self.fuse = nn.Sequential(nn.Linear(in_dim, d_model), nn.LayerNorm(d_model), nn.Dropout(0.1))
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=512,
                                           dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.classifier = nn.Sequential(nn.Linear(d_model, 128), nn.ReLU(), nn.Linear(128, num_classes))
        self.id_head = nn.Linear(d_model, vocab_size)
        self.byte_head = nn.Linear(d_model, 256)
        self.ipc = nn.Sequential(nn.Linear(d_id + 128, 64), nn.ReLU(), nn.Linear(64, 1))

    def encode(self, can_id, payload, time_features, neighbors, weights):
        ident = self.id_emb(can_id)
        pay = self.payload_enc(payload)
        parts = [ident, pay]
        if self.use_time and self.time_enc is not None:
            parts.append(self.time_enc(time_features))
        if self.use_transition and self.trans_enc is not None:
            parts.append(self.trans_enc(neighbors, weights))
        x = self.fuse(torch.cat(parts, dim=-1))
        return self.encoder(x)

    def forward(self, batch: dict) -> torch.Tensor:
        h = self.encode(batch["can_id"], batch["payload"], batch["time_features"],
                        batch["neighbors"], batch["weights"])
        return self.classifier(h.mean(1))

    def pretrain_forward(self, batch: dict) -> dict:
        h = self.encode(batch["can_id"], batch["payload"], batch["time_features"],
                        batch["neighbors"], batch["weights"])
        out = {}
        if self.use_mfm:
            out["id_logits"] = self.id_head(h)
            out["byte_logits"] = self.byte_head(h).unsqueeze(-2).expand(-1, -1, 8, -1)
        if self.use_ipc:
            ident = self.id_emb(batch["can_id"])
            pay = self.payload_enc(batch["payload"])
            out["ipc_logit"] = self.ipc(torch.cat([ident, pay], -1)).squeeze(-1).mean(1)
        return out


def build_model(variant: str = "full") -> SCSCAN:
    if variant == "full":
        return SCSCAN(use_time=True, use_transition=True, use_mfm=True, use_ipc=True)
    if variant == "wo_ssl":
        return SCSCAN(use_time=True, use_transition=True, use_mfm=False, use_ipc=False)
    if variant == "wo_mfm":
        return SCSCAN(use_time=True, use_transition=True, use_mfm=False, use_ipc=True)
    if variant == "wo_ipc":
        return SCSCAN(use_time=True, use_transition=True, use_mfm=True, use_ipc=False)
    if variant == "wo_transition":
        return SCSCAN(use_time=True, use_transition=False, use_mfm=True, use_ipc=True)
    raise ValueError(variant)
