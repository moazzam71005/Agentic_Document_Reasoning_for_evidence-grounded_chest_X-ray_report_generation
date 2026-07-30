"""Strict, non-destructive image validation for chest X-ray views."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError


class UnreadableImageError(ValueError):
    """Raised when a view cannot be decoded into a usable image."""


@dataclass
class ValidatedImage:
    image: np.ndarray
    validation: dict[str, Any]
    checksum: str


def _blur_score(image: np.ndarray) -> float:
    try:
        import cv2

        return float(cv2.Laplacian(image.astype(np.uint8), cv2.CV_64F).var())
    except ImportError:
        if min(image.shape) < 3:
            return 0.0
        gx = np.diff(image.astype(np.float32), axis=1)
        gy = np.diff(image.astype(np.float32), axis=0)
        return float((gx.var() + gy.var()) / 2.0)


def _border_fraction(image: np.ndarray, config: dict[str, Any]) -> float:
    h, w = image.shape
    border_width = max(
        1, round(min(h, w) * float(config.get("border_width_fraction", 0.03)))
    )
    border = np.concatenate(
        [
            image[:border_width, :].ravel(),
            image[-border_width:, :].ravel(),
            image[:, :border_width].ravel(),
            image[:, -border_width:].ravel(),
        ]
    )
    low = int(config.get("saturation_low_value", 1))
    high = int(config.get("saturation_high_value", 254))
    return float(np.mean((border <= low) | (border >= high)))


def _possible_inversion(image: np.ndarray, minimum_delta: float) -> bool:
    h, w = image.shape
    center = image[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
    corners = np.concatenate(
        [
            image[: h // 5, : w // 5].ravel(),
            image[: h // 5, -w // 5 :].ravel(),
            image[-h // 5 :, : w // 5].ravel(),
            image[-h // 5 :, -w // 5 :].ravel(),
        ]
    )
    return bool(corners.mean() > center.mean() + minimum_delta)


def validate_image(
    path: str | Path,
    config: dict[str, Any],
) -> ValidatedImage:
    """Decode, normalize to 8-bit grayscale, and calculate quality warnings."""
    image_path = Path(path)
    if not image_path.exists():
        raise UnreadableImageError(f"Image file does not exist: {image_path}")
    try:
        raw = image_path.read_bytes()
        with Image.open(image_path) as opened:
            original_mode = opened.mode
            bit_depth = 16 if "16" in original_mode or original_mode == "I" else 8
            grayscale = np.asarray(opened.convert("L"), dtype=np.uint8)
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        raise UnreadableImageError(f"Image could not be decoded: {image_path}") from exc
    if grayscale.ndim != 2 or grayscale.size == 0:
        raise UnreadableImageError(f"Decoded image is empty: {image_path}")

    height, width = grayscale.shape
    dynamic_range = float(grayscale.max() - grayscale.min())
    saturation = float(
        np.mean(
            (grayscale <= int(config.get("saturation_low_value", 1)))
            | (grayscale >= int(config.get("saturation_high_value", 254)))
        )
    )
    blur = _blur_score(grayscale)
    border = _border_fraction(grayscale, config)
    possible_inversion = _possible_inversion(
        grayscale, float(config.get("inversion_min_corner_delta", 25.0))
    )
    blank = dynamic_range < float(config["minimum_dynamic_range"])
    warnings: list[str] = []
    if width < int(config["minimum_width"]) or height < int(config["minimum_height"]):
        warnings.append("dimensions_below_minimum")
    if blank:
        warnings.append("low_dynamic_range")
    if saturation > float(config["maximum_saturation_fraction"]):
        warnings.append("excessive_saturation")
    if border > float(config["maximum_border_fraction"]):
        warnings.append("excessive_border")
    if blur < float(config["minimum_blur_score"]):
        warnings.append("extreme_blur")
    if possible_inversion:
        warnings.append("possible_inversion")

    severe = blank or "dimensions_below_minimum" in warnings
    status = "poor" if severe else ("limited" if warnings else "acceptable")
    penalty = min(
        0.8,
        float(config.get("warning_penalty", 0.12)) * len(warnings)
        + (float(config.get("severe_penalty", 0.35)) if severe else 0.0),
    )
    validation = {
        "readable": True,
        "width": int(width),
        "height": int(height),
        "bit_depth": bit_depth,
        "original_mode": original_mode,
        "dynamic_range": dynamic_range,
        "blank_image": blank,
        "possible_inversion": possible_inversion,
        "blur_score": blur,
        "border_fraction": border,
        "saturation_fraction": saturation,
        "severe_clipping": saturation > float(config["maximum_saturation_fraction"]),
        "quality_status": status,
        "quality_score": float(max(0.0, 1.0 - penalty)),
        "warnings": warnings,
    }
    return ValidatedImage(
        image=grayscale,
        validation=validation,
        checksum=hashlib.sha256(raw).hexdigest(),
    )
