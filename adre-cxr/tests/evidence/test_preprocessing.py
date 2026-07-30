import numpy as np

from src.evidence.preprocessing import map_bbox_to_original, preprocess_image

CONFIG = {
    "target_size": 128,
    "normalize_method": "percentile",
    "lower_percentile": 1,
    "upper_percentile": 99,
    "auto_inversion": {"enabled": True},
    "border_crop": {"enabled": False, "threshold_fraction": 0.02},
    "clahe": {"enabled": True, "clip_limit": 2.0, "tile_grid_size": [8, 8]},
}


def test_aspect_ratio_is_preserved_with_padding():
    image = np.tile(np.arange(80, dtype=np.uint8), (40, 1))
    result = preprocess_image(image, CONFIG)
    assert result.normalized_original.shape == (128, 128)
    assert result.transform["resized_width"] == 128
    assert result.transform["resized_height"] == 64
    assert result.transform["padding"]["top"] == 32


def test_bbox_mapping_is_bounded():
    image = np.tile(np.arange(80, dtype=np.uint8), (40, 1))
    result = preprocess_image(image, CONFIG)
    bbox = map_bbox_to_original([0.0, 0.0, 1.0, 1.0], result.transform)
    assert bbox == [0, 0, 80, 40]


def test_preprocessing_is_deterministic():
    rng = np.random.default_rng(42)
    image = rng.integers(0, 256, (70, 100), dtype=np.uint8)
    first = preprocess_image(image, CONFIG)
    second = preprocess_image(image, CONFIG)
    assert np.array_equal(first.normalized_original, second.normalized_original)
    assert np.array_equal(first.clahe_enhanced, second.clahe_enhanced)
    assert first.transform == second.transform
