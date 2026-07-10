"""Dev/test split — freeze after creation."""

from __future__ import annotations

import json
import random
from pathlib import Path


def make_splits(
    records: list[dict],
    dev_size: int = 100,
    test_size: int = 400,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    shuffled = records.copy()
    rng.shuffle(shuffled)
    dev = shuffled[:dev_size]
    test = shuffled[dev_size : dev_size + test_size]
    return dev, test


def save_splits(
    dev: list[dict],
    test: list[dict],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "dev_split.json").write_text(
        json.dumps(dev, indent=2), encoding="utf-8"
    )
    (output_dir / "test_split.json").write_text(
        json.dumps(test, indent=2), encoding="utf-8"
    )
