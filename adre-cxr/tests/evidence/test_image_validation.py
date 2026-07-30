from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.evidence.image_validator import UnreadableImageError, validate_image

CONFIG = {
    "minimum_width": 32,
    "minimum_height": 32,
    "minimum_dynamic_range": 20.0,
    "maximum_saturation_fraction": 0.30,
    "maximum_border_fraction": 0.50,
    "minimum_blur_score": 0.0,
}


def test_grayscale_and_rgb_are_readable(tmp_path: Path):
    gradient = np.tile(np.arange(64, dtype=np.uint8), (64, 1)) * 4
    gray_path = tmp_path / "gray.png"
    rgb_path = tmp_path / "rgb.png"
    Image.fromarray(gradient).save(gray_path)
    Image.fromarray(np.stack([gradient] * 3, axis=-1)).save(rgb_path)
    assert validate_image(gray_path, CONFIG).image.shape == (64, 64)
    assert validate_image(rgb_path, CONFIG).image.shape == (64, 64)


def test_blank_and_small_image_are_marked(tmp_path: Path):
    path = tmp_path / "blank.png"
    Image.fromarray(np.zeros((16, 16), dtype=np.uint8)).save(path)
    result = validate_image(path, CONFIG)
    assert result.validation["blank_image"]
    assert "dimensions_below_minimum" in result.validation["warnings"]


def test_corrupted_image_raises(tmp_path: Path):
    path = tmp_path / "bad.png"
    path.write_bytes(b"not an image")
    with pytest.raises(UnreadableImageError):
        validate_image(path, CONFIG)
