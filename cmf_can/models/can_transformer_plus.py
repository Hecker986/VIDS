"""Transformer rescue models for CAN IDS.

These models deliberately avoid ID-context features. They keep the strong
frame-level sequence backbone and add CAN-specific inductive bias from the
existing frame/window features.
"""
from __future__ import annotations

import math

import torch
from torch import nn


ID_SIZE = 4098


class AttentionPool(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.score = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(h).squeeze(-1), dim=-1)
        return (h * weights.unsqueeze(-1)).sum(dim=1)


class TopKPool(nn.Module):
    def __init__(self, d_model: int, k: int = 5):
        super().__init__()
        self.k = k
        self.score = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        score = self.score(h).squeeze(-1)
        k = min(self.k, score.shape[1])
        idx = torch.topk(score, k=k, dim=1).indices
        gather = idx.unsqueeze(-1).expand(-1, -1, h.shape[-1])
        return h.gather(1, gather).mean(dim=1)


class RelativeTimeTransformerLayer(nn.Module):
    def __init__(self, d_model: int = 256, nhead: int = 4, ff_dim: int = 512, dropout: float = 0.1, buckets: int = 32):
        super().__init__()
        self.nhead = nhead
        self.buckets = buckets
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.rel_time_bias = nn.Embedding(buckets, nhead)
        self.same_id_bias = nn.Parameter(torch.zeros(nhead))
        self.rarity_bias = nn.Parameter(torch.zeros(nhead))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )

    def _bias(self, batch: dict, device: torch.device) -> torch.Tensor:
        can_id = batch["can_id"]
        numeric = batch["frame_numeric"]
        dt_global = numeric[..., 1].abs().nan_to_num(0.0).clamp(max=10.0)
        tpos = torch.cumsum(dt_global, dim=1)
        dist = (tpos[:, :, None] - tpos[:, None, :]).abs()
        bucket = torch.clamp((torch.log1p(dist) * 8).long(), 0, self.buckets - 1)
        bias = self.rel_time_bias(bucket).permute(0, 3, 1, 2)

        same_id = (can_id[:, :, None] == can_id[:, None, :]).float()
        bias = bias + same_id.unsqueeze(1) * self.same_id_bias.view(1, self.nhead, 1, 1)

        rarity = numeric[..., 8].nan_to_num(0.0).clamp(min=0.0, max=20.0)
        pair_rarity = 0.5 * (rarity[:, :, None] + rarity[:, None, :])
        bias = bias + pair_rarity.unsqueeze(1) * self.rarity_bias.view(1, self.nhead, 1, 1) / 20.0
        return bias.reshape(can_id.shape[0] * self.nhead, can_id.shape[1], can_id.shape[1]).to(device)

    def forward(self, x: torch.Tensor, batch: dict, use_bias: bool) -> torch.Tensor:
        h = self.norm1(x)
        attn_mask = self._bias(batch, x.device) if use_bias else None
        a, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + a
        return x + self.ff(self.norm2(x))


class CANTransformerPlus(nn.Module):
    def __init__(
        self,
        numeric_dim: int = 10,
        d_model: int = 256,
        layers: int = 3,
        nhead: int = 4,
        num_classes: int = 2,
        use_same_id_features: bool = True,
        use_relative_time_bias: bool = False,
        pooling: str = "mean",
    ):
        super().__init__()
        self.use_same_id_features = use_same_id_features
        self.use_relative_time_bias = use_relative_time_bias
        self.id_emb = nn.Embedding(ID_SIZE, 64)
        self.byte_emb = nn.Embedding(256, 16)
        self.byte_pos = nn.Embedding(8, 16)
        self.dlc_emb = nn.Embedding(9, 16)
        self.payload_proj = nn.Sequential(nn.Linear(8 * 32, 128), nn.GELU())
        num_dim = numeric_dim if use_same_id_features else 7
        self.num_mlp = nn.Sequential(nn.Linear(num_dim, 96), nn.LayerNorm(96), nn.GELU())
        self.in_proj = nn.Linear(64 + 16 + 128 + 96, d_model)
        self.layers = nn.ModuleList(
            [RelativeTimeTransformerLayer(d_model=d_model, nhead=nhead) for _ in range(layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        if pooling == "attention":
            self.pool = AttentionPool(d_model)
        elif pooling == "topk":
            self.pool = TopKPool(d_model)
        elif pooling == "cls":
            self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
            self.pool = None
        elif pooling == "mean":
            self.pool = None
        else:
            raise ValueError(f"unknown pooling: {pooling}")
        self.pooling = pooling
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128, num_classes))

    def _numeric(self, batch: dict) -> torch.Tensor:
        x = batch["frame_numeric"]
        if self.use_same_id_features:
            return x
        keep = [0, 1, 4, 5, 6, 7, 8]
        return x[..., keep]

    def _payload(self, payload: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(8, device=payload.device).view(1, 1, 8)
        emb = torch.cat([self.byte_emb(payload), self.byte_pos(pos).expand(payload.shape[0], payload.shape[1], -1, -1)], dim=-1)
        return self.payload_proj(emb.flatten(-2))

    def encode(self, batch: dict) -> torch.Tensor:
        payload = batch["payload"]
        dlc = (batch["frame_numeric"][..., 0].clamp(0, 1) * 8).round().long().clamp(0, 8)
        x = torch.cat(
            [
                self.id_emb(batch["can_id"]),
                self.dlc_emb(dlc),
                self._payload(payload),
                self.num_mlp(self._numeric(batch)),
            ],
            dim=-1,
        )
        h = self.in_proj(x)
        if self.pooling == "cls":
            cls = self.cls.expand(h.shape[0], -1, -1)
            h = torch.cat([cls, h], dim=1)
            local_batch = dict(batch)
            local_batch["can_id"] = torch.cat([batch["can_id"][:, :1], batch["can_id"]], dim=1)
            local_batch["frame_numeric"] = torch.cat([batch["frame_numeric"][:, :1] * 0, batch["frame_numeric"]], dim=1)
        else:
            local_batch = batch
        for layer in self.layers:
            h = layer(h, local_batch, self.use_relative_time_bias)
        h = self.norm(h)
        if self.pooling == "cls":
            z = h[:, 0]
        elif self.pool is not None:
            z = self.pool(h)
        else:
            z = h.mean(dim=1)
        self.last_embedding = z
        return z

    def forward(self, batch: dict, return_aux: bool = False):
        z = self.encode(batch)
        logits = self.head(z)
        return (logits, {}) if return_aux else logits


class TemporalFrameStatsTransformer(nn.Module):
    def __init__(
        self,
        fusion: str = "concat",
        numeric_dim: int = 10,
        stats_dim: int = 26,
        d_model: int = 256,
        num_classes: int = 2,
    ):
        super().__init__()
        self.fusion = fusion
        self.frame = CANTransformerPlus(numeric_dim=numeric_dim, d_model=d_model, use_same_id_features=True, use_relative_time_bias=False)
        self.stats = nn.Sequential(
            nn.Linear(stats_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        if fusion == "concat":
            head_in = d_model * 2
            self.gate = None
            self.cross = None
        elif fusion == "gate":
            head_in = d_model
            self.gate = nn.Sequential(nn.Linear(d_model * 2, 128), nn.GELU(), nn.Linear(128, d_model), nn.Sigmoid())
            self.cross = None
        elif fusion == "stats_attention":
            head_in = d_model
            self.gate = None
            layer = nn.TransformerEncoderLayer(d_model, 4, 512, dropout=0.1, activation="gelu", batch_first=True, norm_first=True)
            self.cross = nn.TransformerEncoder(layer, num_layers=1)
            self.type_emb = nn.Parameter(torch.zeros(2, d_model))
        else:
            raise ValueError(f"unknown fusion: {fusion}")
        self.head = nn.Sequential(nn.LayerNorm(head_in), nn.Linear(head_in, 128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128, num_classes))

    def forward(self, batch: dict, return_aux: bool = False):
        z_frame = self.frame.encode(batch)
        z_stats = self.stats(batch["window_stats"])
        if self.fusion == "concat":
            z = torch.cat([z_frame, z_stats], dim=-1)
        elif self.fusion == "gate":
            gate = self.gate(torch.cat([z_frame, z_stats], dim=-1))
            z = z_frame + gate * z_stats
        else:
            tokens = torch.stack([z_frame, z_stats], dim=1) + self.type_emb.unsqueeze(0)
            z = self.cross(tokens).mean(dim=1)
        self.last_embedding = z
        logits = self.head(z)
        return (logits, {}) if return_aux else logits


def make_can_transformer_plus(name: str) -> nn.Module:
    if name == "can_transformer_plus_basic":
        return CANTransformerPlus(use_same_id_features=False, use_relative_time_bias=False)
    if name == "can_transformer_plus_sameid":
        return CANTransformerPlus(use_same_id_features=True, use_relative_time_bias=False)
    if name == "can_transformer_plus_timebias":
        return CANTransformerPlus(use_same_id_features=False, use_relative_time_bias=True)
    if name == "can_transformer_plus_sameid_timebias":
        return CANTransformerPlus(use_same_id_features=True, use_relative_time_bias=True)
    if name == "can_transformer_plus_attnpool":
        return CANTransformerPlus(use_same_id_features=True, use_relative_time_bias=False, pooling="attention")
    if name == "can_transformer_plus_topk":
        return CANTransformerPlus(use_same_id_features=True, use_relative_time_bias=False, pooling="topk")
    if name == "tfscan_concat":
        return TemporalFrameStatsTransformer(fusion="concat")
    if name == "tfscan_gate":
        return TemporalFrameStatsTransformer(fusion="gate")
    if name == "tfscan_stats_attention":
        return TemporalFrameStatsTransformer(fusion="stats_attention")
    raise ValueError(f"unknown transformer rescue model: {name}")
