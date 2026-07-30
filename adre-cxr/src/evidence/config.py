"""Configuration loading and reproducibility helpers for evidence v2."""

from __future__ import annotations

import hashlib
import json
import random
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def load_evidence_config(path: str | Path) -> dict[str, Any]:
    """Load an evidence configuration and perform basic structural checks."""
    config_path = Path(path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"Evidence config must be a mapping: {config_path}")
    required = {
        "schema_version",
        "extractor_version",
        "preprocessing",
        "classification",
        "threshold_defaults",
        "fusion",
        "output",
    }
    missing = sorted(required - data.keys())
    if missing:
        raise ValueError(f"Missing evidence config keys: {', '.join(missing)}")
    return data


def config_hash(config: dict[str, Any]) -> str:
    """Return a stable SHA-256 hash for a parsed configuration."""
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def set_deterministic_seed(seed: int) -> None:
    """Seed supported random-number generators without requiring PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


def apply_ablation(config: dict[str, Any], ablation: int) -> dict[str, Any]:
    """Return a config for the controlled evidence ablations (1 through 7)."""
    if ablation not in range(1, 8):
        raise ValueError("Evidence ablation must be an integer from 1 to 7")
    result = deepcopy(config)
    result["views"]["use_all_available"] = ablation >= 3
    result["localization"]["enabled"] = ablation >= 4
    result["anatomy"]["enabled"] = ablation >= 5
    result["region_analysis"]["enabled"] = ablation >= 5
    result["embeddings"]["enabled"] = ablation >= 6
    if ablation == 1:
        result["thresholds"] = {}
        result["threshold_defaults"] = {
            "absent_max": 0.50,
            "probably_absent_max": 0.50,
            "uncertain_max": 0.50,
            "present_min": 0.50,
        }
    if ablation < 7:
        for index, model in enumerate(result["classification"]["models"]):
            model["enabled"] = index == 0
    else:
        for model in result["classification"]["models"]:
            model["enabled"] = True
    result["ablation"] = {
        "number": ablation,
        "description": {
            1: "quality_validation_and_deterministic_preprocessing",
            2: "ablation_1_plus_uncertainty_statuses",
            3: "ablation_2_plus_multi_view",
            4: "ablation_3_plus_gradcam",
            5: "ablation_4_plus_heuristic_anatomy_and_regions",
            6: "ablation_5_plus_dense_embeddings",
            7: "ablation_6_plus_model_ensemble",
        }[ablation],
    }
    return result
