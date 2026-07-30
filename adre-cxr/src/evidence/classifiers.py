"""Pluggable visual evidence models with a TorchXRayVision implementation."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image

LOGGER = logging.getLogger(__name__)


@dataclass
class ModelPrediction:
    model_name: str
    model_version: str
    labels: dict[str, float]
    embedding: np.ndarray | None = None
    feature_maps: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class EvidenceModel(ABC):
    name: str
    version: str

    @abstractmethod
    def predict(self, image: np.ndarray, view_type: str) -> ModelPrediction:
        """Predict model scores from a square 8-bit grayscale image."""

    def attribution(self, image: np.ndarray, label: str) -> np.ndarray | None:
        """Return a normalized model-attention map, if supported."""
        return None


class TorchXRayVisionModel(EvidenceModel):
    """DenseNet wrapper exposing scores, pooled features, and Grad-CAM."""

    def __init__(
        self,
        weights: str = "densenet121-res224-all",
        device: str | None = None,
        name: str = "torchxrayvision_densenet",
    ):
        try:
            import torch
            import torchxrayvision as xrv
        except ImportError as exc:
            raise ImportError(
                "TorchXRayVision model requested but torch/torchxrayvision is unavailable"
            ) from exc
        self._torch = torch
        self._xrv = xrv
        self.name = name
        self.version = weights
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = xrv.models.DenseNet(weights=weights).to(self.device)
        self.model.eval()
        self.pathologies = list(self.model.pathologies)
        self._activation = None
        self._gradient = None

    def _tensor(self, image: np.ndarray):
        normalized = self._xrv.datasets.normalize(image.astype(np.float32), 255)
        return (
            self._torch.from_numpy(normalized)
            .unsqueeze(0)
            .unsqueeze(0)
            .float()
            .to(self.device)
        )

    def _feature_module(self):
        features = getattr(self.model, "features", None)
        if features is None:
            return None
        children = list(features.children())
        return children[-1] if children else features

    def predict(self, image: np.ndarray, view_type: str) -> ModelPrediction:
        tensor = self._tensor(image)
        activation: list[Any] = []
        module = self._feature_module()
        handle = None
        if module is not None:
            handle = module.register_forward_hook(
                lambda _module, _inputs, output: activation.append(output.detach())
            )
        try:
            with self._torch.no_grad():
                output = self.model(tensor)[0]
        finally:
            if handle is not None:
                handle.remove()
        labels = {
            pathology: float(output[index].detach().cpu())
            for index, pathology in enumerate(self.pathologies)
        }
        embedding = None
        if activation:
            pooled = activation[-1].mean(dim=(-2, -1)).squeeze(0)
            embedding = pooled.detach().cpu().numpy().astype(np.float32)
        return ModelPrediction(
            model_name=self.name,
            model_version=self.version,
            labels=labels,
            embedding=embedding,
            metadata={"device": self.device, "view_type": view_type},
        )

    def attribution(self, image: np.ndarray, label: str) -> np.ndarray | None:
        if label not in self.pathologies:
            return None
        module = self._feature_module()
        if module is None:
            return None
        activation: list[Any] = []
        gradient: list[Any] = []

        def forward_hook(_module, _inputs, output):
            activation.append(output)
            output.register_hook(lambda grad: gradient.append(grad))

        handle = module.register_forward_hook(forward_hook)
        tensor = self._tensor(image)
        tensor.requires_grad_(True)
        try:
            self.model.zero_grad(set_to_none=True)
            output = self.model(tensor)[0]
            output[self.pathologies.index(label)].backward()
        finally:
            handle.remove()
        if not activation or not gradient:
            return None
        weights = gradient[-1].mean(dim=(-2, -1), keepdim=True)
        heatmap = self._torch.relu((weights * activation[-1]).sum(dim=1))[0]
        heatmap = heatmap.detach().cpu().numpy()
        if float(heatmap.max()) <= float(heatmap.min()):
            return None
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        return np.asarray(
            Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
                (image.shape[1], image.shape[0]), Image.Resampling.BILINEAR
            ),
            dtype=np.float32,
        ) / 255.0


def build_models(config: dict[str, Any], device: str | None = None) -> list[tuple[EvidenceModel, float]]:
    """Build configured model plugins and their ensemble weights."""
    models: list[tuple[EvidenceModel, float]] = []
    for item in config.get("models", []):
        if not item.get("enabled", True):
            continue
        name = item["name"]
        if name == "torchxrayvision_densenet":
            model = TorchXRayVisionModel(
                weights=item["weights"],
                device=device,
                name=item.get("id", name),
            )
        else:
            raise ValueError(f"Unsupported evidence model plugin: {name}")
        models.append((model, float(item.get("weight", 1.0))))
    return models
