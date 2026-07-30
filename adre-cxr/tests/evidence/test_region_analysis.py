import numpy as np

from src.evidence.region_analysis import analyze_fixed_grid


def test_legacy_fixed_grid_is_preserved():
    image = np.arange(120 * 80, dtype=np.float32).reshape(120, 80)
    regions = analyze_fixed_grid(image, rows=3, cols=2)
    assert len(regions) == 6
    assert regions[0]["region"] == "upper_left"
    assert regions[-1]["region"] == "lower_right"
    assert all(
        0 <= value <= 1
        for region in regions
        for value in region["bbox_normalized"]
    )
