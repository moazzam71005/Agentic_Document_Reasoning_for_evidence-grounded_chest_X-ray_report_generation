"""Conservative quantitative measurements with explicit assessability."""

from __future__ import annotations

from typing import Any

import numpy as np


def _maximum_width(mask: np.ndarray) -> int:
    if not mask.any():
        return 0
    return int(max(row.sum() for row in mask))


def cardiothoracic_ratio(
    masks: dict[str, np.ndarray],
    view_type: str,
    anatomy_method: str,
) -> dict[str, Any]:
    """Estimate CTR only on PA views; never claim clinical-grade segmentation."""
    if view_type != "frontal_pa":
        return {
            "value": None,
            "method": "not_computed",
            "assessable": False,
            "confidence": 0.0,
            "view_requirement_met": False,
            "interpretation": "not_assessable",
            "limitations": ["CTR is only interpreted on PA projections."],
        }
    heart = masks.get("heart")
    left = masks.get("left_lung")
    right = masks.get("right_lung")
    if heart is None or left is None or right is None:
        return {
            "value": None,
            "method": "not_computed",
            "assessable": False,
            "confidence": 0.0,
            "view_requirement_met": True,
            "interpretation": "not_assessable",
            "limitations": ["Required heart/lung masks are unavailable."],
        }
    heart_width = _maximum_width(heart)
    thorax = left | right | heart
    thorax_width = _maximum_width(thorax)
    if thorax_width <= 0:
        value = None
    else:
        value = float(heart_width / thorax_width)
    interpretation = (
        "not_assessable"
        if value is None
        else "enlarged"
        if value > 0.55
        else "borderline_enlarged"
        if value > 0.50
        else "not_enlarged"
    )
    heuristic = anatomy_method.startswith("heuristic")
    return {
        "value": value,
        "method": f"{anatomy_method}_ctr",
        "assessable": value is not None,
        "confidence": 0.30 if heuristic else 0.80,
        "view_requirement_met": True,
        "interpretation": interpretation,
        "limitations": (
            [
                "Experimental estimate from geometric masks; do not use as clinical truth."
            ]
            if heuristic
            else []
        ),
    }


def lung_asymmetry(image: np.ndarray, masks: dict[str, np.ndarray]) -> dict[str, Any]:
    left, right = masks.get("left_lung"), masks.get("right_lung")
    if left is None or right is None or not left.any() or not right.any():
        return {
            "value": None,
            "method": "mask_intensity_difference",
            "assessable": False,
            "confidence": 0.0,
            "limitations": ["Both lung masks are required."],
        }
    left_mean, right_mean = float(image[left].mean()), float(image[right].mean())
    difference = abs(left_mean - right_mean) / max(1.0, left_mean, right_mean)
    return {
        "value": float(difference),
        "method": "heuristic_mask_intensity_difference",
        "assessable": True,
        "confidence": 0.35,
        "limitations": [
            "Sensitive to rotation, exposure, and heuristic mask placement."
        ],
    }
