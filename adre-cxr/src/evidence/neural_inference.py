"""Pretrained CXR classifier wrapper — no fine-tuning required."""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import torch
    import torchvision
    import torchxrayvision as xrv
    HAS_TXRV = True
except ImportError:
    HAS_TXRV = False


class NeuralInference:
    """Global pathology probabilities via TorchXRayVision DenseNet."""

    def __init__(self, weights: str = "densenet121-res224-nih"):
        if not HAS_TXRV:
            raise ImportError(
                "torchxrayvision not installed. "
                "Install on Kaggle/Colab GPU, or use precomputed evidence JSON."
            )
        self.model = xrv.models.DenseNet(weights=weights)
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.pathologies = list(self.model.pathologies)
        self.transform = torchvision.transforms.Compose([
            xrv.datasets.XRayCenterCrop(),
            xrv.datasets.XRayResizer(224),
        ])

    def _prepare_tensor(self, img: np.ndarray) -> "torch.Tensor":
        if img.ndim == 3:
            img = img.mean(axis=-1)
        img_norm = xrv.datasets.normalize(img, 255)
        if img_norm.ndim == 2:
            img_norm = img_norm[None, ...]
        img_ready = self.transform(img_norm)
        t = torch.from_numpy(img_ready).unsqueeze(0)
        if torch.cuda.is_available():
            t = t.cuda()
        return t

    def predict(self, img: np.ndarray) -> dict[str, float]:
        """img: 2D grayscale float array (0–255)."""
        t = self._prepare_tensor(img)
        with torch.no_grad():
            probs = torch.sigmoid(self.model(t))[0].cpu().numpy()
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
