"""Deterministic image features: CTR proxy, opacity density, symmetry."""

from __future__ import annotations

import numpy as np


def opacity_density(img: np.ndarray, threshold_pct: float = 60.0) -> float:
    """Fraction of pixels above intensity threshold (opacity proxy)."""
    thresh = np.percentile(img, threshold_pct)
    return float(np.mean(img > thresh))


def symmetry_index(left: np.ndarray, right: np.ndarray) -> float:
    """L/R symmetry: 1.0 = identical histograms, 0.0 = maximally different."""
    if left.size == 0 or right.size == 0:
        return 0.0
    h_l, _ = np.histogram(left.flatten(), bins=32, range=(0, 255), density=True)
    h_r, _ = np.histogram(right.flatten(), bins=32, range=(0, 255), density=True)
    diff = np.abs(h_l - h_r).sum()
    return float(max(0.0, 1.0 - diff / 2.0))


def split_left_right(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mid = img.shape[1] // 2
    return img[:, :mid], img[:, mid:]


def cardiothoracic_ratio_proxy(img: np.ndarray) -> float:
    """
    Rough CTR proxy from horizontal mass distribution.
    True CTR needs heart/lung segmentation; this is a fast heuristic.
    """
    h, _ = img.shape
    row_means = img.mean(axis=1)
    center = h // 2
    upper = row_means[:center].mean() if center > 0 else 0
    lower = row_means[center:].mean() if center < h else 0
    return float(lower / (upper + 1e-6))


def extract_features(img: np.ndarray) -> dict[str, float]:
    left, right = split_left_right(img)
    return {
        "opacity_density": opacity_density(img),
        "symmetry_index": symmetry_index(left, right),
        "ctr_proxy": cardiothoracic_ratio_proxy(img),
        "mean_intensity": float(img.mean()),
        "std_intensity": float(img.std()),
    }
