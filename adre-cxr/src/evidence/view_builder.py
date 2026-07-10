"""Image view generation: crops, zones, contrast enhancement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from PIL import Image


@dataclass
class ImageView:
    view_id: str
    image: np.ndarray
    params: dict[str, Any]
    tool: str = "view_builder"


def load_grayscale(path: str) -> np.ndarray:
    img = np.array(Image.open(path).convert("L"))
    return img.astype(np.float32)


def clahe_enhance(img: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    u8 = np.clip(img, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    return clahe.apply(u8).astype(np.float32)


def simple_lung_bbox(img: np.ndarray, margin: float = 0.05) -> tuple[int, int, int, int]:
    """Heuristic lung bounding box when no segmentation model is available."""
    h, w = img.shape[:2]
    ys, xs = np.where(img > np.percentile(img, 20))
    if len(xs) == 0:
        return 0, 0, h, w
    y0, y1 = max(0, ys.min()), min(h, ys.max())
    x0, x1 = max(0, xs.min()), min(w, xs.max())
    dy = int((y1 - y0) * margin)
    dx = int((x1 - x0) * margin)
    return max(0, y0 - dy), min(h, y1 + dy), max(0, x0 - dx), min(w, x1 + dx)


def zone_crops(img: np.ndarray, rows: int = 3, cols: int = 2) -> list[ImageView]:
    y0, y1, x0, x1 = simple_lung_bbox(img)
    lung = img[y0:y1, x0:x1]
    zh, zw = lung.shape[0] // rows, lung.shape[1] // cols
    zone_names = [
        ("upper", "left"), ("upper", "right"),
        ("mid", "left"), ("mid", "right"),
        ("lower", "left"), ("lower", "right"),
    ]
    views: list[ImageView] = []
    idx = 0
    for r in range(rows):
        for c in range(cols):
            zy0, zy1 = r * zh, (r + 1) * zh if r < rows - 1 else lung.shape[0]
            zx0, zx1 = c * zw, (c + 1) * zw if c < cols - 1 else lung.shape[1]
            crop = lung[zy0:zy1, zx0:zx1]
            vrow, vcol = zone_names[idx]
            views.append(ImageView(
                view_id=f"zone_{vrow}_{vcol}",
                image=crop,
                params={"row": vrow, "col": vcol, "bbox": [y0, y1, x0, x1]},
                tool="zone_crop",
            ))
            idx += 1
    return views


def build_views(
    image_path: str,
    rows: int = 3,
    cols: int = 2,
    include_clahe: bool = True,
) -> list[ImageView]:
    """Generate all callable views for one chest X-ray."""
    original = load_grayscale(image_path)
    views = [
        ImageView(view_id="original", image=original, params={}, tool="load"),
    ]
    if include_clahe:
        views.append(ImageView(
            view_id="clahe",
            image=clahe_enhance(original),
            params={"clip_limit": 2.0},
            tool="clahe",
        ))
    views.extend(zone_crops(original, rows=rows, cols=cols))
    return views
