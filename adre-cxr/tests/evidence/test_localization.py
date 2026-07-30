import numpy as np

from src.evidence.localization import regions_from_heatmap

TRANSFORM = {
    "target_size": 100,
    "scale": 1.0,
    "padding": {"left": 0, "right": 0, "top": 0, "bottom": 0},
    "crop_bbox_original_xyxy": [0, 0, 100, 100],
    "original_size": {"width": 100, "height": 100},
}


def test_no_activation_has_no_regions():
    regions = regions_from_heatmap(np.zeros((100, 100)), TRANSFORM, 0.5, 0.01, 3, 0.3)
    assert regions == []


def test_multiple_activation_regions_are_bounded():
    heatmap = np.zeros((100, 100), dtype=np.float32)
    heatmap[10:30, 10:30] = 0.8
    heatmap[60:90, 60:90] = 0.9
    regions = regions_from_heatmap(heatmap, TRANSFORM, 0.5, 0.01, 3, 0.3)
    assert len(regions) == 2
    assert all(
        0 <= value <= 1 for region in regions for value in region["bbox_normalized"]
    )
