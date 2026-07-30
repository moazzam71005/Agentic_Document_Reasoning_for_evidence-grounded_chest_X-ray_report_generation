"""Model-attention localization and normalized region extraction."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.evidence.preprocessing import map_bbox_to_original


def _connected_components(binary: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    try:
        import cv2

        count, _, stats, _ = cv2.connectedComponentsWithStats(
            binary.astype(np.uint8), connectivity=8
        )
        return [
            (
                int(stats[i, cv2.CC_STAT_LEFT]),
                int(stats[i, cv2.CC_STAT_TOP]),
                int(stats[i, cv2.CC_STAT_WIDTH]),
                int(stats[i, cv2.CC_STAT_HEIGHT]),
                int(stats[i, cv2.CC_STAT_AREA]),
            )
            for i in range(1, count)
        ]
    except ImportError:
        # Deterministic flood fill used in lightweight test environments.
        visited = np.zeros(binary.shape, dtype=bool)
        components: list[tuple[int, int, int, int, int]] = []
        height, width = binary.shape
        for y in range(height):
            for x in range(width):
                if not binary[y, x] or visited[y, x]:
                    continue
                stack = [(y, x)]
                visited[y, x] = True
                points: list[tuple[int, int]] = []
                while stack:
                    cy, cx = stack.pop()
                    points.append((cy, cx))
                    for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                        if (
                            0 <= ny < height
                            and 0 <= nx < width
                            and binary[ny, nx]
                            and not visited[ny, nx]
                        ):
                            visited[ny, nx] = True
                            stack.append((ny, nx))
                ys, xs = zip(*points)
                components.append(
                    (min(xs), min(ys), max(xs) - min(xs) + 1, max(ys) - min(ys) + 1, len(points))
                )
        return components


def _iou(a: list[float], b: list[float]) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return intersection / max(1e-8, area_a + area_b - intersection)


def regions_from_heatmap(
    heatmap: np.ndarray,
    transform: dict[str, Any],
    threshold: float,
    min_area_fraction: float,
    max_regions: int,
    merge_iou_threshold: float,
) -> list[dict[str, Any]]:
    """Convert a normalized attention map to inspectable candidate regions."""
    if heatmap.size == 0 or float(heatmap.max()) <= 0:
        return []
    binary = heatmap >= float(threshold)
    image_area = float(heatmap.shape[0] * heatmap.shape[1])
    candidates: list[dict[str, Any]] = []
    for x, y, width, height, area in _connected_components(binary):
        if area / image_area < min_area_fraction:
            continue
        bbox = [
            x / heatmap.shape[1],
            y / heatmap.shape[0],
            (x + width) / heatmap.shape[1],
            (y + height) / heatmap.shape[0],
        ]
        activation = float(heatmap[y : y + height, x : x + width].mean())
        candidates.append(
            {
                "bbox_normalized": [float(max(0.0, min(1.0, value))) for value in bbox],
                "bbox_original_xyxy": map_bbox_to_original(bbox, transform),
                "activation_score": activation,
            }
        )
    candidates.sort(key=lambda item: item["activation_score"], reverse=True)
    retained: list[dict[str, Any]] = []
    for candidate in candidates:
        if any(
            _iou(candidate["bbox_normalized"], item["bbox_normalized"])
            >= merge_iou_threshold
            for item in retained
        ):
            continue
        retained.append(candidate)
        if len(retained) >= max_regions:
            break
    return retained


def assign_anatomical_label(
    bbox: list[float], left_lung_mask: np.ndarray, right_lung_mask: np.ndarray
) -> dict[str, str]:
    """Assign side and vertical lung zone by mask overlap."""
    height, width = left_lung_mask.shape
    x0, y0, x1, y1 = [
        round(bbox[0] * width),
        round(bbox[1] * height),
        round(bbox[2] * width),
        round(bbox[3] * height),
    ]
    x0, x1 = max(0, x0), min(width, max(x0 + 1, x1))
    y0, y1 = max(0, y0), min(height, max(y0 + 1, y1))
    left_overlap = int(left_lung_mask[y0:y1, x0:x1].sum())
    right_overlap = int(right_lung_mask[y0:y1, x0:x1].sum())
    side = "left" if left_overlap > right_overlap else "right"
    if left_overlap == right_overlap == 0:
        side = "indeterminate"
    center_y = (y0 + y1) / 2.0 / max(1, height)
    zone = "upper" if center_y < 1 / 3 else ("middle" if center_y < 2 / 3 else "lower")
    return {
        "side": side,
        "vertical_zone": zone,
        "anatomical_region": f"{side}_{zone}_lung",
    }
