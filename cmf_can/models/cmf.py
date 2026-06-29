"""CMF-CAN and baselines."""
from __future__ import annotations

import torch
from torch import nn


class FrameEncoder(nn.Module):
    def __init__(self, numeric_dim: int = 10, d_model: int = 256, layers: int = 3, id_dropout: float = 0.0):
        super().__init__()
        self.id_dropout = id_dropout
        self.id_emb = nn.Embedding(4098, 64)
        self.byte_emb = nn.Embedding(256, 16)
        self.payload_proj = nn.Sequential(nn.Linear(128, 128), nn.GELU())
        self.num_mlp = nn.Sequential(nn.Linear(numeric_dim, 64), nn.LayerNorm(64), nn.GELU())
        self.in_proj = nn.Linear(64 + 128 + 64, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=512,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(d_model)

    def _payload(self, payload: torch.Tensor) -> torch.Tensor:
        return self.payload_proj(self.byte_emb(payload).flatten(-2))

    def forward(self, batch: dict) -> torch.Tensor:
        can_id = batch["can_id"]
        if self.training and self.id_dropout > 0:
            mask = torch.rand_like(can_id.float()) < self.id_dropout
            can_id = can_id.masked_fill(mask, 4097)
        x = torch.cat(
            [
                self.id_emb(can_id),
                self._payload(batch["payload"]),
                self.num_mlp(batch["frame_numeric"]),
            ],
            dim=-1,
        )
        h = self.encoder(self.in_proj(x))
        return self.norm(h.mean(dim=1))


class StatsEncoder(nn.Module):
    def __init__(self, stats_dim: int = 26, d_model: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(stats_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

    def forward(self, batch: dict) -> torch.Tensor:
        return self.net(batch["window_stats"])


class ContextEncoder(nn.Module):
    def __init__(self, context_dim: int = 18, d_model: int = 256):
        super().__init__()
        self.per_frame = nn.Sequential(
            nn.Linear(context_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, d_model),
            nn.GELU(),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, batch: dict) -> torch.Tensor:
        h = self.per_frame(batch["id_context"])
        return self.norm(h.mean(dim=1))


class CMFCAN(nn.Module):
    def __init__(
        self,
        stats_dim: int = 26,
        context_dim: int = 18,
        numeric_dim: int = 10,
        d_model: int = 256,
        num_classes: int = 2,
        use_stats: bool = True,
        use_context: bool = True,
        use_xattn: bool = True,
        use_gate: bool = True,
        concat_only: bool = False,
        frame_only: bool = False,
        stats_only: bool = False,
        id_dropout: float = 0.0,
        modality_dropout: float = 0.0,
    ):
        super().__init__()
        self.frame_only = frame_only
        self.stats_only = stats_only
        self.use_stats = use_stats or stats_only
        self.use_context = use_context and not stats_only
        self.use_xattn = use_xattn and not concat_only and not frame_only and not stats_only
        self.use_gate = use_gate and not concat_only and not frame_only and not stats_only
        self.concat_only = concat_only
        self.modality_dropout = modality_dropout
        self.active_aux_indices = [0]
        if self.use_stats:
            self.active_aux_indices.append(1)
        if self.use_context:
            self.active_aux_indices.append(2)
        self.frame = FrameEncoder(numeric_dim=numeric_dim, d_model=d_model, id_dropout=id_dropout)
        self.stats = StatsEncoder(stats_dim=stats_dim, d_model=d_model)
        self.context = ContextEncoder(context_dim=context_dim, d_model=d_model)
        self.modality_type = nn.Parameter(torch.zeros(3, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=512,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.cross = nn.TransformerEncoder(layer, num_layers=1)
        self.gate = nn.Sequential(nn.Linear(d_model * 3, 128), nn.GELU(), nn.Linear(128, 3))
        self.frame_head = nn.Linear(d_model, num_classes)
        self.stats_head = nn.Linear(d_model, num_classes)
        self.context_head = nn.Linear(d_model, num_classes)
        self.use_logit_residual = not concat_only and not frame_only and not stats_only
        head_in = d_model
        if concat_only:
            head_in = d_model * 3
        self.head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes),
        )

    def _tokens(self, batch: dict) -> torch.Tensor:
        z_frame = self.frame(batch)
        z_stats = self.stats(batch) if self.use_stats else torch.zeros_like(z_frame)
        z_ctx = self.context(batch) if self.use_context else torch.zeros_like(z_frame)
        return torch.stack([z_frame, z_stats, z_ctx], dim=1)

    def forward(self, batch: dict, return_aux: bool = False):
        aux: dict[str, torch.Tensor] = {}
        if self.stats_only:
            h = self.stats(batch)
            self.last_embedding = h
            logits = self.head(h)
            return (logits, aux) if return_aux else logits
        tokens = self._tokens(batch)
        self.last_aux_logits = (
            self.frame_head(tokens[:, 0]),
            self.stats_head(tokens[:, 1]),
            self.context_head(tokens[:, 2]),
        )
        if self.training and self.modality_dropout > 0 and not self.frame_only:
            keep = (torch.rand(tokens.shape[:2], device=tokens.device) >= self.modality_dropout).float()
            keep[:, 1] = 1.0
            empty = keep.sum(dim=1) == 0
            keep[empty, 1] = 1.0
            tokens = tokens * keep.unsqueeze(-1)
        if self.frame_only:
            self.last_embedding = tokens[:, 0]
            logits = self.head(tokens[:, 0])
            return (logits, aux) if return_aux else logits
        if self.concat_only:
            h = tokens.flatten(1)
            self.last_embedding = h
            logits = self.head(h)
            return (logits, aux) if return_aux else logits
        if self.use_xattn:
            tokens = self.cross(tokens + self.modality_type.unsqueeze(0))
        if self.use_gate:
            weights = torch.softmax(self.gate(tokens.flatten(1)), dim=-1).unsqueeze(-1)
            fused = (tokens * weights).sum(dim=1)
            aux = {
                "gate_frame": weights[:, 0, 0],
                "gate_window": weights[:, 1, 0],
                "gate_context": weights[:, 2, 0],
            }
        else:
            fused = tokens.mean(dim=1)
        self.last_embedding = fused
        logits = self.head(fused)
        if self.use_logit_residual:
            logits = logits + 0.5 * self.last_aux_logits[0]
        return (logits, aux) if return_aux else logits


class SequenceBaseline(nn.Module):
    def __init__(self, kind: str, numeric_dim: int = 10, hidden: int = 256, num_classes: int = 2):
        super().__init__()
        self.kind = kind
        self.id_emb = nn.Embedding(4098, 64)
        self.byte_emb = nn.Embedding(256, 16)
        in_dim = 64 + 128 + numeric_dim
        self.pay_proj = nn.Sequential(nn.Linear(128, 128), nn.GELU())
        if kind == "cnn":
            self.backbone = nn.Sequential(
                nn.Conv1d(in_dim, 128, 3, padding=1), nn.GELU(),
                nn.Conv1d(128, hidden, 3, padding=1), nn.GELU(),
                nn.AdaptiveAvgPool1d(1),
            )
        elif kind == "lstm":
            self.in_proj = nn.Linear(in_dim, hidden)
            self.backbone = nn.LSTM(hidden, hidden, num_layers=2, dropout=0.1, batch_first=True)
        elif kind == "gru":
            self.in_proj = nn.Linear(in_dim, hidden)
            self.backbone = nn.GRU(hidden, hidden, num_layers=2, dropout=0.1, batch_first=True)
        elif kind == "transformer":
            self.in_proj = nn.Linear(in_dim, hidden)
            layer = nn.TransformerEncoderLayer(hidden, 4, 512, dropout=0.1, activation="gelu", batch_first=True, norm_first=True)
            self.backbone = nn.TransformerEncoder(layer, num_layers=3)
        else:
            raise ValueError(kind)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 128), nn.GELU(), nn.Linear(128, num_classes))

    def _input(self, batch: dict) -> torch.Tensor:
        pay = self.pay_proj(self.byte_emb(batch["payload"]).flatten(-2))
        return torch.cat([self.id_emb(batch["can_id"]), pay, batch["frame_numeric"]], dim=-1)

    def forward(self, batch: dict) -> torch.Tensor:
        x = self._input(batch)
        if self.kind == "cnn":
            h = self.backbone(x.transpose(1, 2)).squeeze(-1)
        elif self.kind in {"lstm", "gru"}:
            h, _ = self.backbone(self.in_proj(x))
            h = h.mean(dim=1)
        else:
            h = self.backbone(self.in_proj(x)).mean(dim=1)
        return self.head(h)


def build_model(name: str) -> nn.Module:
    if name in {"cnn", "lstm", "gru", "transformer"}:
        return SequenceBaseline(name)
    if name == "frame_only":
        return CMFCAN(frame_only=True)
    if name == "stats_only":
        return CMFCAN(stats_only=True)
    if name == "concat_fusion":
        return CMFCAN(concat_only=True)
    if name == "cmf_can":
        return CMFCAN()
    if name == "cmf_can_supcon":
        return CMFCAN(id_dropout=0.2, modality_dropout=0.1)
    if name == "cmf_can_robust":
        return CMFCAN(id_dropout=0.4, modality_dropout=0.2)
    if name == "wo_stats":
        return CMFCAN(use_stats=False)
    if name == "wo_context":
        return CMFCAN(use_context=False)
    if name == "wo_xattn":
        return CMFCAN(use_xattn=False)
    if name == "wo_gate":
        return CMFCAN(use_gate=False)
    if name in {"reliable_cmf_can", "reliable_cmf_can_no_shift", "reliable_cmf_can_no_segment"}:
        from cmf_can.models.reliable_cmf_can import ReliableCMFCAN

        return ReliableCMFCAN(
            use_shift_control=name != "reliable_cmf_can_no_shift",
            use_segment_pooling=name != "reliable_cmf_can_no_segment",
        )
    raise ValueError(f"unknown model: {name}")
