"""Reliable-CMF-CAN model prototype.

This module adds reliability-aware gating, shift-aware context control, and
sparse segment evidence without modifying the original CMF-CAN class.
"""
from __future__ import annotations

import torch
from torch import nn

from cmf_can.models.cmf import ContextEncoder, FrameEncoder, StatsEncoder
from cmf_can.models.reliability_gate import ReliabilityGate
from cmf_can.models.segment_pooling import SegmentTopKPooling


class ReliableCMFCAN(nn.Module):
    def __init__(
        self,
        stats_dim: int = 26,
        context_dim: int = 18,
        numeric_dim: int = 10,
        d_model: int = 256,
        num_classes: int = 2,
        reliability_hidden_dim: int = 128,
        modality_dropout: float = 0.1,
        use_shift_control: bool = True,
        use_segment_pooling: bool = True,
    ):
        super().__init__()
        self.use_shift_control = use_shift_control
        self.use_segment_pooling = use_segment_pooling
        self.modality_dropout = modality_dropout
        self.frame = FrameEncoder(numeric_dim=numeric_dim, d_model=d_model, id_dropout=0.1)
        self.stats = StatsEncoder(stats_dim=stats_dim, d_model=d_model)
        self.context = ContextEncoder(context_dim=context_dim, d_model=d_model)
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
        self.modality_type = nn.Parameter(torch.zeros(3, d_model))
        self.reliability_gate = ReliabilityGate(d_model=d_model, hidden_dim=reliability_hidden_dim, shift_dim=7)
        self.segment_pool = SegmentTopKPooling(numeric_dim=numeric_dim)
        self.frame_head = nn.Linear(d_model, num_classes)
        self.stats_head = nn.Linear(d_model, num_classes)
        self.context_head = nn.Linear(d_model, num_classes)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes),
        )
        self.segment_head = nn.Linear(1, num_classes)
        self.active_aux_indices = [0, 1, 2]

    def _shift_features(self, batch: dict, z_context: torch.Tensor) -> torch.Tensor:
        id_context = batch["id_context"]
        context_abs = id_context.abs().sum(dim=-1)
        unknown_id_ratio = (context_abs <= 1e-6).float().mean(dim=1, keepdim=True)
        context_distance = z_context.norm(dim=-1, keepdim=True) / (z_context.shape[-1] ** 0.5)
        stats = batch["window_stats"]
        stat_mean = stats.mean(dim=-1, keepdim=True)
        stat_std = stats.std(dim=-1, keepdim=True)
        stat_max = stats.abs().amax(dim=-1, keepdim=True)
        dt_mean = batch["frame_numeric"][..., 1].mean(dim=1, keepdim=True)
        payload_delta = batch["frame_numeric"][..., -1].abs().mean(dim=1, keepdim=True)
        return torch.cat([unknown_id_ratio, context_distance, stat_mean, stat_std, stat_max, dt_mean, payload_delta], dim=-1)

    def forward(self, batch: dict, return_aux: bool = False):
        z_frame = self.frame(batch)
        z_window = self.stats(batch)
        z_context = self.context(batch)
        shift_features = self._shift_features(batch, z_context)
        if self.use_shift_control:
            context_mask = (1.0 - shift_features[:, :1]).clamp(0.0, 1.0)
            z_context = z_context * context_mask
        else:
            context_mask = torch.ones_like(shift_features[:, :1])
        tokens = torch.stack([z_frame, z_window, z_context], dim=1)
        self.last_aux_logits = (
            self.frame_head(tokens[:, 0]),
            self.stats_head(tokens[:, 1]),
            self.context_head(tokens[:, 2]),
        )
        if self.training and self.modality_dropout > 0:
            keep = (torch.rand(tokens.shape[:2], device=tokens.device) >= self.modality_dropout).float()
            keep[:, 1] = 1.0
            tokens = tokens * keep.unsqueeze(-1)
        tokens = self.cross(tokens + self.modality_type.unsqueeze(0))
        reliability, weights = self.reliability_gate(tokens, shift_features)
        fused = (tokens * weights.unsqueeze(-1)).sum(dim=1)
        self.last_embedding = fused
        logits = self.head(fused) + 0.5 * self.last_aux_logits[0]
        segment_scores = None
        topk_idx = None
        topk_score = None
        if self.use_segment_pooling:
            topk_score, segment_scores, topk_idx = self.segment_pool(batch["frame_numeric"])
            logits = logits + 0.2 * self.segment_head(topk_score.unsqueeze(-1))
        aux = {
            "gate_frame": weights[:, 0],
            "gate_window": weights[:, 1],
            "gate_context": weights[:, 2],
            "reliability_frame": reliability[:, 0],
            "reliability_window": reliability[:, 1],
            "reliability_context": reliability[:, 2],
            "context_shift_score": shift_features[:, 0],
            "context_mask_value": context_mask[:, 0],
        }
        if segment_scores is not None and topk_score is not None and topk_idx is not None:
            aux["topk_score"] = topk_score
            aux["segment_scores"] = segment_scores
            aux["topk_segment_indices"] = topk_idx.float()
        return (logits, aux) if return_aux else logits
