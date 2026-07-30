"""Score calibration and conservative evidence-status interpretation."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from src.evidence.schemas import FindingStatus


class ScoreCalibrator:
    """Apply pre-fitted temperature or Platt parameters without fitting on test data."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.method = str(config.get("method", "none")).lower()
        self.calibrated = bool(config.get("calibrated", False))
        self.parameters: dict[str, Any] = {}
        parameter_path = config.get("parameters_path")
        if self.calibrated:
            if not parameter_path:
                raise ValueError("calibrated=true requires calibration.parameters_path")
            self.parameters = json.loads(
                Path(parameter_path).read_text(encoding="utf-8")
            )
            if self.method not in {"temperature", "platt"}:
                raise ValueError(f"Unsupported calibration method: {self.method}")

    @staticmethod
    def _logit(score: float) -> float:
        clipped = min(1.0 - 1e-6, max(1e-6, score))
        return math.log(clipped / (1.0 - clipped))

    def apply(self, label: str, raw_score: float) -> float:
        if not self.calibrated or self.method == "none":
            return float(raw_score)
        params = self.parameters.get(label, self.parameters.get("default", {}))
        logit = self._logit(float(raw_score))
        if self.method == "temperature":
            temperature = float(params.get("temperature", 1.0))
            if temperature <= 0:
                raise ValueError(f"Invalid temperature for {label}: {temperature}")
            value = logit / temperature
        else:
            value = float(params.get("slope", 1.0)) * logit + float(
                params.get("intercept", 0.0)
            )
        return float(1.0 / (1.0 + math.exp(-value)))

    def metadata(self) -> dict[str, Any]:
        return {
            "method": self.method if self.calibrated else "none",
            "calibrated": self.calibrated,
            "calibration_split": self.config.get("calibration_split"),
            "calibrator_version": self.config.get("calibrator_version"),
            "confidence_type": (
                "calibrated_probability" if self.calibrated else "heuristic_model_score"
            ),
        }


def interpret_score(
    label: str,
    score: float,
    thresholds: dict[str, dict[str, float]],
    defaults: dict[str, float],
    assessable: bool = True,
) -> str:
    """Map a score to a six-state evidence interpretation."""
    if not assessable:
        return FindingStatus.NOT_ASSESSABLE.value
    values = {**defaults, **thresholds.get(label, {})}
    ordered = [
        values["absent_max"],
        values["probably_absent_max"],
        values["uncertain_max"],
        values["present_min"],
    ]
    if ordered != sorted(ordered):
        raise ValueError(f"Thresholds for {label} are not monotonic")
    if score < values["absent_max"]:
        return FindingStatus.ABSENT.value
    if score < values["probably_absent_max"]:
        return FindingStatus.PROBABLY_ABSENT.value
    if score < values["uncertain_max"]:
        return FindingStatus.UNCERTAIN.value
    if score < values["present_min"]:
        return FindingStatus.PROBABLY_PRESENT.value
    return FindingStatus.PRESENT.value
