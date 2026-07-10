"""Quality indicators: rotation, exposure, inspiration heuristics."""

from __future__ import annotations

import numpy as np


def exposure_score(img: np.ndarray) -> dict[str, float | str]:
    """Penetration/exposure heuristic from intensity distribution."""
    p5, p95 = np.percentile(img, [5, 95])
    dynamic_range = p95 - p5
    if dynamic_range < 80:
        label = "underexposed"
    elif dynamic_range > 220:
        label = "overexposed"
    else:
        label = "adequate"
    return {"dynamic_range": float(dynamic_range), "exposure_label": label}


def inspiration_proxy(img: np.ndarray) -> dict[str, float | str]:
    """Rib-count proxy via vertical edge density in lung field."""
    grad = np.abs(np.diff(img, axis=0))
    edge_density = float(grad.mean())
    if edge_density > 12:
        label = "adequate"
    elif edge_density > 8:
        label = "suboptimal"
    else:
        label = "poor"
    return {"rib_edge_density": edge_density, "inspiration_label": label}


def rotation_proxy(img: np.ndarray) -> dict[str, float | str]:
    """Clavicle symmetry proxy via L/R upper-quadrant intensity balance."""
    h, w = img.shape
    upper = img[: h // 4, :]
    left = upper[:, : w // 2].mean()
    right = upper[:, w // 2 :].mean()
    imbalance = abs(left - right) / (max(left, right) + 1e-6)
    if imbalance < 0.1:
        label = "minimal"
    elif imbalance < 0.25:
        label = "mild"
    else:
        label = "moderate"
    return {"clavicle_imbalance": float(imbalance), "rotation_label": label}


def assess_quality(img: np.ndarray) -> dict:
    return {
        **exposure_score(img),
        **inspiration_proxy(img),
        **rotation_proxy(img),
    }
