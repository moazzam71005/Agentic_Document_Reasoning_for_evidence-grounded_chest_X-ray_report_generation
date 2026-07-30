"""Heuristic anatomy masks used until a validated segmentation model is supplied."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def _ellipse_mask(
    shape: tuple[int, int], center: tuple[float, float], radii: tuple[float, float]
) -> np.ndarray:
    height, width = shape
    yy, xx = np.ogrid[:height, :width]
    cx, cy = center
    rx, ry = radii
    return (((xx - cx) / max(rx, 1.0)) ** 2 + ((yy - cy) / max(ry, 1.0)) ** 2 <= 1.0)


def approximate_anatomy(
    image: np.ndarray,
    view_type: str,
    config: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Return explicitly heuristic lung/heart/mediastinum masks."""
    height, width = image.shape
    if view_type not in {"frontal_pa", "frontal_ap"}:
        empty = np.zeros_like(image, dtype=bool)
        return (
            {
                "method": "not_available_for_projection",
                "segmentation_quality": 0.0,
                "limitations": ["Anatomical masks are only approximated for frontal views."],
            },
            {"left_lung": empty, "right_lung": empty, "heart": empty, "mediastinum": empty},
        )

    geometry = (config or {}).get("geometry", {})
    left_center = geometry.get("left_lung_center", [0.34, 0.51])
    right_center = geometry.get("right_lung_center", [0.66, 0.51])
    lung_radii = geometry.get("lung_radii", [0.20, 0.36])
    heart_center = geometry.get("heart_center", [0.52, 0.64])
    heart_radii = geometry.get("heart_radii", [0.19, 0.17])
    mediastinum_box = geometry.get("mediastinum_box", [0.44, 0.20, 0.58, 0.70])
    # These patient-relative ellipses are safer than full-image rectangular
    # zones but are not a clinical segmentation.
    left = _ellipse_mask(
        image.shape,
        (width * left_center[0], height * left_center[1]),
        (width * lung_radii[0], height * lung_radii[1]),
    )
    right = _ellipse_mask(
        image.shape,
        (width * right_center[0], height * right_center[1]),
        (width * lung_radii[0], height * lung_radii[1]),
    )
    heart = _ellipse_mask(
        image.shape,
        (width * heart_center[0], height * heart_center[1]),
        (width * heart_radii[0], height * heart_radii[1]),
    )
    mediastinum = np.zeros_like(image, dtype=bool)
    mediastinum[
        int(height * mediastinum_box[1]) : int(height * mediastinum_box[3]),
        int(width * mediastinum_box[0]) : int(width * mediastinum_box[2]),
    ] = True
    masks = {"left_lung": left, "right_lung": right, "heart": heart, "mediastinum": mediastinum}
    return (
        {
            "method": "heuristic_geometric_masks",
            "segmentation_quality": 0.35,
            "limitations": [
                "Masks are geometric approximations, not model-derived segmentations.",
                "Measurements derived from these masks are experimental.",
            ],
        },
        masks,
    )


def save_masks(masks: dict[str, np.ndarray], directory: Path) -> dict[str, str]:
    directory.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for name, mask in masks.items():
        path = directory / f"{name}_mask.png"
        Image.fromarray(mask.astype(np.uint8) * 255).save(path)
        paths[f"{name}_mask"] = str(path)
    return paths


def save_anatomy_overlay(
    image: np.ndarray, masks: dict[str, np.ndarray], path: Path
) -> None:
    """Save a color overlay that makes heuristic masks auditable."""
    base = np.stack([image] * 3, axis=-1).astype(np.uint8)
    colors = {
        "left_lung": np.array([0, 180, 255], dtype=np.uint8),
        "right_lung": np.array([0, 255, 100], dtype=np.uint8),
        "heart": np.array([255, 60, 60], dtype=np.uint8),
        "mediastinum": np.array([255, 180, 0], dtype=np.uint8),
    }
    for name, mask in masks.items():
        if not mask.any():
            continue
        color = colors[name]
        base[mask] = (0.75 * base[mask] + 0.25 * color).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(base).save(path)
