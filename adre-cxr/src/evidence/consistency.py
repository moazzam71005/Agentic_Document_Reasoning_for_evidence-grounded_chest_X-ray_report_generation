"""Rule-based evidence consistency and anatomical plausibility checks."""

from __future__ import annotations

from typing import Any


def run_consistency_checks(
    views: list[dict[str, Any]],
    groups: dict[str, list[dict[str, Any]]],
    availability: dict[str, bool],
    model_disagreement_threshold: float = 0.35,
) -> list[dict[str, str]]:
    checks: list[dict[str, str]] = []
    all_findings = [item for group in groups.values() for item in group]
    for finding in all_findings:
        if finding.get("view_agreement") == "conflicting":
            checks.append(
                {
                    "type": "view_conflict",
                    "finding": finding["name"],
                    "severity": "warning",
                    "message": "Selected view scores differ beyond the configured conflict threshold.",
                }
            )
        if finding.get("reliability_level") == "very_low":
            checks.append(
                {
                    "type": "low_reliability",
                    "finding": finding["name"],
                    "severity": "warning",
                    "message": "Evidence reliability is very low; do not state this finding.",
                }
            )

    cardiomegaly = next(
        (
            item
            for item in groups["positive"]
            if item["name"].lower() == "cardiomegaly"
        ),
        None,
    )
    if cardiomegaly:
        frontal_anatomy = [
            view.get("anatomy", {})
            for view in views
            if view.get("view_type") in {"frontal_pa", "frontal_ap"}
        ]
        if not frontal_anatomy or max(
            [item.get("segmentation_quality", 0.0) for item in frontal_anatomy] or [0.0]
        ) <= 0:
            checks.append(
                {
                    "type": "missing_anatomical_support",
                    "finding": "Cardiomegaly",
                    "severity": "warning",
                    "message": "Cardiomegaly score has no heart-region support.",
                }
            )
    if not availability.get("lateral"):
        checks.append(
            {
                "type": "missing_view",
                "finding": "study",
                "severity": "info",
                "message": "Lateral view unavailable; do not claim confirmation on both views.",
            }
        )

    for view in views:
        if view.get("view_type") == "frontal_ap":
            checks.append(
                {
                    "type": "measurement_limitation",
                    "finding": "Cardiothoracic ratio",
                    "severity": "info",
                    "message": "AP projection can magnify the heart; strong CTR interpretation is disabled.",
                }
            )
        for localization in view.get("localizations", []):
            for region in localization.get("regions", []):
                if region.get("side") == "indeterminate":
                    checks.append(
                        {
                            "type": "localization_outside_lung",
                            "finding": localization.get("finding", "unknown"),
                            "severity": "warning",
                            "message": "Model-attended region does not overlap an approximate lung mask.",
                        }
                    )
            if localization.get("finding", "").lower() in {"effusion", "pleural effusion"}:
                for region in localization.get("regions", []):
                    if region.get("vertical_zone") == "upper":
                        checks.append(
                            {
                                "type": "anatomical_inconsistency",
                                "finding": localization["finding"],
                                "severity": "warning",
                                "message": "Model-attended effusion region is upper-zone and anatomically atypical.",
                            }
                        )
        for prediction in view.get("global_predictions", []):
            if (
                float(prediction.get("model_disagreement", 0.0))
                >= model_disagreement_threshold
            ):
                checks.append(
                    {
                        "type": "model_conflict",
                        "finding": prediction.get("label", "unknown"),
                        "severity": "warning",
                        "message": "Classifier ensemble members disagree beyond the configured threshold.",
                    }
                )
    return checks
