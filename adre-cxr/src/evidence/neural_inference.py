"""Pretrained CXR classifier wrapper — no fine-tuning required."""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import torch
    import torchxrayvision as xrv
    HAS_TXRV = True
except ImportError:
    HAS_TXRV = False


class NeuralInference:
    """Global pathology probabilities via TorchXRayVision DenseNet."""

    def __init__(self, weights: str = "densenet121-res224-all"):
        if not HAS_TXRV:
            raise ImportError(
                "torchxrayvision not installed. "
                "Install on Kaggle/Colab GPU, or use precomputed evidence JSON."
            )
        self.model = xrv.models.DenseNet(weights=weights)
        self.model.eval()
        self.pathologies = list(self.model.pathologies)

    @staticmethod
    def _center_crop_square(img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        side = min(h, w)
        y0 = (h - side) // 2
        x0 = (w - side) // 2
        return img[y0 : y0 + side, x0 : x0 + side]

    def predict(self, img: np.ndarray) -> dict[str, float]:
        """img: 2D grayscale float array."""
        img = self._center_crop_square(img)
        img_norm = xrv.datasets.normalize(img.astype(np.float32), 255)
        tensor = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            probs = self.model(tensor)[0].numpy()
        return {p: float(probs[i]) for i, p in enumerate(self.pathologies)}

    def global_abnormality_score(self, probs: dict[str, float]) -> float:
        """Aggregate pathology probs into single P_ab (max probability)."""
        if not probs:
            return 0.0
        return max(probs.values())

    def regional_burden(
        self,
        zone_probs: dict[str, dict[str, float]],
    ) -> dict[str, dict[str, float]]:
        """
        Per-zone pathology burden — analogue of ADRE Eq. 1 for images.
        zone_probs: {zone_id: {pathology: prob}}
        """
        burden: dict[str, dict[str, float]] = {}
        for zone_id, probs in zone_probs.items():
            burden[zone_id] = {
                "max_pathology_prob": max(probs.values()) if probs else 0.0,
                "mean_pathology_prob": float(np.mean(list(probs.values()))) if probs else 0.0,
                "top_pathology": max(probs, key=probs.get) if probs else "none",
            }
        return burden
