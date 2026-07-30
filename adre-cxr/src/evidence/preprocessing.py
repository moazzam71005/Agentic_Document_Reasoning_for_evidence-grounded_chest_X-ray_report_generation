"""Deterministic aspect-ratio-preserving chest X-ray preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageOps


@dataclass
class PreprocessedImage:
    normalized_original: np.ndarray
    clahe_enhanced: np.ndarray | None
    transform: dict[str, Any]


def _content_bbox(
    image: np.ndarray, border_config: dict[str, Any]
) -> tuple[int, int, int, int]:
    low, high = np.percentile(
        image,
        [
            float(border_config.get("background_lower_percentile", 1.0)),
            float(border_config.get("background_upper_percentile", 99.0)),
        ],
    )
    spread = max(1.0, float(high - low))
    threshold = max(
        float(border_config.get("minimum_foreground_delta", 2.0)),
        spread * float(border_config["threshold_fraction"]),
    )
    background = float(np.median(np.concatenate([image[0], image[-1], image[:, 0], image[:, -1]])))
    foreground = np.abs(image.astype(np.float32) - background) > threshold
    ys, xs = np.where(foreground)
    if len(xs) == 0:
        return 0, 0, image.shape[1], image.shape[0]
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def _normalize_percentile(
    image: np.ndarray, lower_percentile: float, upper_percentile: float
) -> np.ndarray:
    lower, upper = np.percentile(image, [lower_percentile, upper_percentile])
    if upper <= lower:
        return np.zeros_like(image, dtype=np.uint8)
    scaled = (image.astype(np.float32) - float(lower)) / float(upper - lower)
    return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)


def _resize_and_pad(image: np.ndarray, target_size: int) -> tuple[np.ndarray, dict[str, Any]]:
    height, width = image.shape
    scale = target_size / max(height, width)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    resized = np.asarray(
        Image.fromarray(image).resize((resized_width, resized_height), Image.Resampling.BILINEAR)
    )
    pad_left = (target_size - resized_width) // 2
    pad_right = target_size - resized_width - pad_left
    pad_top = (target_size - resized_height) // 2
    pad_bottom = target_size - resized_height - pad_top
    padded = np.pad(
        resized,
        ((pad_top, pad_bottom), (pad_left, pad_right)),
        mode="constant",
        constant_values=0,
    )
    return padded, {
        "scale": float(scale),
        "resized_width": resized_width,
        "resized_height": resized_height,
        "padding": {
            "left": pad_left,
            "right": pad_right,
            "top": pad_top,
            "bottom": pad_bottom,
        },
    }


def _clahe(image: np.ndarray, clip_limit: float, tile_grid_size: list[int]) -> np.ndarray:
    try:
        import cv2

        processor = cv2.createCLAHE(
            clipLimit=float(clip_limit), tileGridSize=tuple(int(x) for x in tile_grid_size)
        )
        return processor.apply(image)
    except ImportError:
        return np.asarray(ImageOps.equalize(Image.fromarray(image)))


def preprocess_image(
    image: np.ndarray,
    config: dict[str, Any],
    possible_inversion: bool = False,
) -> PreprocessedImage:
    """Return normalized and CLAHE branches plus invertible geometry metadata."""
    original_height, original_width = image.shape
    bbox = (0, 0, original_width, original_height)
    cropped = image
    border_config = config["border_crop"]
    if border_config.get("enabled", True):
        bbox = _content_bbox(image, border_config)
        cropped = image[bbox[1] : bbox[3], bbox[0] : bbox[2]]

    inverted = bool(config["auto_inversion"].get("enabled", True) and possible_inversion)
    if inverted:
        cropped = 255 - cropped
    normalized = _normalize_percentile(
        cropped,
        float(config["lower_percentile"]),
        float(config["upper_percentile"]),
    )
    padded, resize_meta = _resize_and_pad(normalized, int(config["target_size"]))
    clahe_config = config["clahe"]
    enhanced = None
    if clahe_config.get("enabled", True):
        enhanced = _clahe(
            padded,
            float(clahe_config["clip_limit"]),
            list(clahe_config["tile_grid_size"]),
        )
    transform = {
        "method": "percentile_normalize_aspect_resize_pad",
        "original_size": {"width": original_width, "height": original_height},
        "crop_bbox_original_xyxy": list(bbox),
        "auto_inverted": inverted,
        "target_size": int(config["target_size"]),
        **resize_meta,
    }
    return PreprocessedImage(
        normalized_original=padded,
        clahe_enhanced=enhanced,
        transform=transform,
    )


def map_bbox_to_original(
    bbox_normalized: list[float], transform: dict[str, Any]
) -> list[int]:
    """Map an ``[x0,y0,x1,y1]`` normalized padded-image box to source pixels."""
    target = float(transform["target_size"])
    padding = transform["padding"]
    scale = float(transform["scale"])
    crop_x0, crop_y0, _, _ = transform["crop_bbox_original_xyxy"]
    x0 = (bbox_normalized[0] * target - padding["left"]) / scale + crop_x0
    y0 = (bbox_normalized[1] * target - padding["top"]) / scale + crop_y0
    x1 = (bbox_normalized[2] * target - padding["left"]) / scale + crop_x0
    y1 = (bbox_normalized[3] * target - padding["top"]) / scale + crop_y0
    size = transform["original_size"]
    return [
        round(max(0.0, min(float(size["width"]), x0))),
        round(max(0.0, min(float(size["height"]), y0))),
        round(max(0.0, min(float(size["width"]), x1))),
        round(max(0.0, min(float(size["height"]), y1))),
    ]
