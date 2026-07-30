"""Model-ensemble and cross-view evidence fusion with disagreement tracking."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from src.evidence.calibration import ScoreCalibrator, interpret_score


def fuse_model_predictions(
    predictions: list[tuple[dict[str, float], str, str, float]],
    calibrator: ScoreCalibrator,
    thresholds: dict[str, dict[str, float]],
    defaults: dict[str, float],
    assessable: bool,
) -> list[dict[str, Any]]:
    """Fuse ``(scores, model_name, version, weight)`` for one view."""
    labels = sorted({label for item, _, _, _ in predictions for label in item})
    output: list[dict[str, Any]] = []
    for label in labels:
        sources: list[dict[str, Any]] = []
        weighted_values: list[tuple[float, float]] = []
        for scores, model_name, version, weight in predictions:
            if label not in scores:
                continue
            raw = float(scores[label])
            if not np.isfinite(raw):
                continue
            raw = float(max(0.0, min(1.0, raw)))
            interpreted = calibrator.apply(label, raw)
            sources.append(
                {
                    "source": model_name,
                    "model_version": version,
                    "raw_score": raw,
                    "interpreted_score": interpreted,
                    "weight": float(weight),
                }
            )
            weighted_values.append((interpreted, float(weight)))
        if not sources:
            continue
        total_weight = sum(weight for _, weight in weighted_values)
        score = (
            sum(value * weight for value, weight in weighted_values) / total_weight
            if total_weight > 0
            else 0.0
        )
        source_scores = [value for value, _ in weighted_values]
        disagreement = (
            float(max(source_scores) - min(source_scores)) if len(source_scores) > 1 else 0.0
        )
        output.append(
            {
                "label": label,
                "raw_score": float(
                    np.average(
                        [source["raw_score"] for source in sources],
                        weights=[source["weight"] for source in sources],
                    )
                ),
                "interpreted_score": float(score),
                "status": interpret_score(
                    label, float(score), thresholds, defaults, assessable=assessable
                ),
                "calibrated": calibrator.calibrated,
                "confidence_type": calibrator.metadata()["confidence_type"],
                "model_disagreement": disagreement,
                "provenance": sources,
            }
        )
    return output


def _agreement(values: list[float], conflict_threshold: float) -> tuple[str, float]:
    if len(values) < 2:
        return "single_view", 0.5
    spread = max(values) - min(values)
    if spread >= conflict_threshold:
        return "conflicting", max(0.0, 1.0 - spread)
    if spread <= conflict_threshold / 2.0:
        return "agree", min(1.0, 1.0 - spread)
    return "partial", max(0.5, 1.0 - spread)


def fuse_views(
    views: list[dict[str, Any]],
    config: dict[str, Any],
    thresholds: dict[str, dict[str, float]],
    defaults: dict[str, float],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Fuse per-view predictions without erasing view-specific evidence."""
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for view in views:
        if not view.get("selected", True):
            continue
        for prediction in view.get("global_predictions", []):
            by_label[prediction["label"]].append(
                {
                    "view_id": view["view_id"],
                    "view_type": view["view_type"],
                    "score": prediction["interpreted_score"],
                    "status": prediction["status"],
                    "quality_score": view.get("validation", {}).get("quality_score", 0.0),
                    "model_disagreement": prediction.get("model_disagreement", 0.0),
                    "localization_confidence": max(
                        [
                            region.get("activation_score", 0.0)
                            for loc in view.get("localizations", [])
                            if loc.get("finding") == prediction["label"]
                            for region in loc.get("regions", [])
                        ]
                        or [0.0]
                    ),
                    "anatomy_quality": view.get("anatomy", {}).get(
                        "segmentation_quality", 0.0
                    ),
                }
            )

    groups: dict[str, list[dict[str, Any]]] = {
        "positive": [],
        "negative": [],
        "uncertain": [],
        "not_assessable": [],
    }
    cross_view: dict[str, Any] = {"findings": []}
    view_weights = config.get("view_weights", {})
    conflict_threshold = float(config["conflict_threshold"])
    reliability_config = config["reliability"]
    for label, supports in sorted(by_label.items()):
        weighted: list[tuple[float, float]] = []
        for support in supports:
            weight = float(view_weights.get(support["view_type"], 0.5))
            weight *= max(0.05, float(support["quality_score"]))
            weighted.append((float(support["score"]), weight))
        total_weight = sum(weight for _, weight in weighted)
        fused_score = (
            sum(score * weight for score, weight in weighted) / total_weight
            if total_weight
            else 0.0
        )
        agreement, view_agreement = _agreement(
            [score for score, _ in weighted], conflict_threshold
        )
        assessable = any(
            support["status"] != "not_assessable" for support in supports
        )
        status = interpret_score(
            label, fused_score, thresholds, defaults, assessable=assessable
        )
        model_agreement = float(
            1.0 - np.mean([support["model_disagreement"] for support in supports])
        )
        quality_score = float(np.mean([support["quality_score"] for support in supports]))
        localization_score = float(
            max(support["localization_confidence"] for support in supports)
        )
        anatomy_score = float(
            max(support["anatomy_quality"] for support in supports)
        )
        reliability = (
            float(reliability_config["model_agreement_weight"]) * model_agreement
            + float(reliability_config["view_agreement_weight"]) * view_agreement
            + float(reliability_config["image_quality_weight"]) * quality_score
            + float(reliability_config["localization_weight"]) * localization_score
            + float(reliability_config["anatomy_weight"]) * anatomy_score
        )
        reliability = float(max(0.0, min(1.0, reliability)))
        levels = reliability_config.get(
            "levels", {"high_min": 0.80, "moderate_min": 0.60, "low_min": 0.40}
        )
        reliability_level = (
            "high"
            if reliability >= float(levels["high_min"])
            else "moderate"
            if reliability >= float(levels["moderate_min"])
            else "low"
            if reliability >= float(levels["low_min"])
            else "very_low"
        )
        finding = {
            "name": label,
            "status": status,
            "score": float(fused_score),
            "confidence": float(
                fused_score
                if status in {"present", "probably_present"}
                else 1.0 - fused_score
                if status in {"absent", "probably_absent"}
                else 0.5
            ),
            "view_agreement": agreement,
            "reliability_score": reliability,
            "reliability_level": reliability_level,
            "provenance": supports,
        }
        if status in {"present", "probably_present"}:
            groups["positive"].append(finding)
        elif status in {"absent", "probably_absent"}:
            groups["negative"].append(finding)
        elif status == "not_assessable":
            groups["not_assessable"].append(finding)
        else:
            groups["uncertain"].append(finding)
        cross_view["findings"].append(
            {
                "finding": label,
                "per_view": {
                    support["view_id"]: {
                        "view_type": support["view_type"],
                        "score": support["score"],
                        "status": support["status"],
                    }
                    for support in supports
                },
                "fused": {
                    "score": float(fused_score),
                    "status": status,
                    "agreement": agreement,
                },
            }
        )
    return groups, cross_view


def generation_guidance(
    groups: dict[str, list[dict[str, Any]]],
) -> dict[str, list[str]]:
    """Create deterministic phrasing constraints from status and reliability."""
    safe_positive = [
        item["name"]
        for item in groups["positive"]
        if item["status"] == "present" and item["reliability_level"] in {"high", "moderate"}
    ]
    safe_negative = [
        f"No {item['name'].lower()}"
        for item in groups["negative"]
        if item["status"] == "absent" and item["reliability_level"] in {"high", "moderate"}
    ]
    hedged = [
        f"Possible {item['name'].lower()}"
        for item in groups["positive"] + groups["uncertain"]
        if item["name"] not in safe_positive
    ]
    do_not_state = [
        item["name"]
        for item in groups["not_assessable"]
    ] + [
        item["name"]
        for item in groups["positive"] + groups["negative"]
        if item["reliability_level"] == "very_low"
    ]
    return {
        "safe_positive_findings": sorted(set(safe_positive)),
        "safe_negative_findings": sorted(set(safe_negative)),
        "hedged_findings": sorted(set(hedged)),
        "do_not_state": sorted(set(do_not_state)),
    }
