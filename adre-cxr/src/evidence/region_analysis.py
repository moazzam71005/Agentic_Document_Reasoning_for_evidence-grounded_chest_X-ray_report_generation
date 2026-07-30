"""Anatomy-relative deterministic regional evidence."""

from __future__ import annotations

from typing import Any

import numpy as np


def analyze_lung_regions(
    image: np.ndarray, masks: dict[str, np.ndarray]
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for side in ("left", "right"):
        lung = masks.get(f"{side}_lung")
        if lung is None or not lung.any():
            continue
        ys, _ = np.where(lung)
        top, bottom = int(ys.min()), int(ys.max() + 1)
        boundaries = np.linspace(top, bottom, 4).astype(int)
        for index, zone in enumerate(("upper", "middle", "lower")):
            zone_mask = lung.copy()
            zone_mask[: boundaries[index], :] = False
            zone_mask[boundaries[index + 1] :, :] = False
            pixels = image[zone_mask]
            if pixels.size == 0:
                continue
            results.append(
                {
                    "region": f"{side}_{zone}_lung",
                    "method": "heuristic_lung_mask_thirds",
                    "mean_intensity": float(pixels.mean()),
                    "std_intensity": float(pixels.std()),
                    "area_fraction": float(zone_mask.mean()),
                    "assessable": True,
                }
            )
    return results


def compact_visual_embedding(image: np.ndarray, size: list[int]) -> np.ndarray:
    """Deterministic fallback embedding; explicitly not a learned semantic vector."""
    from PIL import Image

    resized = np.asarray(
        Image.fromarray(image.astype(np.uint8)).resize(
            (int(size[1]), int(size[0])), Image.Resampling.BILINEAR
        ),
        dtype=np.float32,
    )
    vector = resized.reshape(-1)
    vector -= vector.mean()
    norm = float(np.linalg.norm(vector))
    if norm > 0:
        vector /= norm
    return vector.astype(np.float32)


def analyze_fixed_grid(
    image: np.ndarray, rows: int, cols: int
) -> list[dict[str, Any]]:
    """Preserve the legacy fixed grid as a separately labelled evidence source."""
    if rows <= 0 or cols <= 0:
        raise ValueError("Fixed-grid rows and columns must be positive")
    height, width = image.shape
    row_names = (
        ["upper", "middle", "lower"]
        if rows == 3
        else [f"row_{index}" for index in range(rows)]
    )
    col_names = (
        ["left", "right"]
        if cols == 2
        else [f"col_{index}" for index in range(cols)]
    )
    regions: list[dict[str, Any]] = []
    for row in range(rows):
        for col in range(cols):
            y0, y1 = round(row * height / rows), round((row + 1) * height / rows)
            x0, x1 = round(col * width / cols), round((col + 1) * width / cols)
            crop = image[y0:y1, x0:x1]
            regions.append(
                {
                    "region": f"{row_names[row]}_{col_names[col]}",
                    "method": "legacy_fixed_full_image_grid",
                    "bbox_normalized": [
                        x0 / width,
                        y0 / height,
                        x1 / width,
                        y1 / height,
                    ],
                    "mean_intensity": float(crop.mean()) if crop.size else 0.0,
                    "std_intensity": float(crop.std()) if crop.size else 0.0,
                    "assessable": bool(crop.size),
                }
            )
    return regions
