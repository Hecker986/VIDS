"""Benign-normality scoring utilities for Reliable-CMF-CAN analysis."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class NormalityBoundary:
    """A small normal-only boundary model over window statistics.

    This class is intentionally numpy-based so it can be fit from processed
    features without coupling the train loop to sklearn objects.
    """

    center: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, normal_features: np.ndarray) -> "NormalityBoundary":
        center = np.median(normal_features, axis=0)
        q25, q75 = np.percentile(normal_features, [25, 75], axis=0)
        scale = np.maximum(q75 - q25, 1e-6)
        return cls(center=center.astype(np.float32), scale=scale.astype(np.float32))

    def score(self, features: np.ndarray) -> np.ndarray:
        z = np.abs((features - self.center) / self.scale)
        return z.mean(axis=1).astype(np.float32)
