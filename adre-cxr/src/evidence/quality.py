"""Study-level quality aggregation."""

from __future__ import annotations

from typing import Any


def aggregate_study_quality(
    views: list[dict[str, Any]], availability: dict[str, bool]
) -> dict[str, Any]:
    selected = [view for view in views if view.get("selected", True)]
    scores = [
        float(view.get("validation", {}).get("quality_score", 0.0)) for view in selected
    ]
    overall = sum(scores) / len(scores) if scores else 0.0
    limitations: list[str] = []
    if not selected:
        limitations.append("No readable views were available.")
    if not availability.get("frontal"):
        limitations.append("No frontal view was available.")
    if not availability.get("lateral"):
        limitations.append(
            "No lateral view was available; cross-view confirmation is limited."
        )
    for view in selected:
        for warning in view.get("validation", {}).get("warnings", []):
            limitations.append(f"{view['view_id']}: {warning}")
    status = (
        "acceptable" if overall >= 0.80 else ("limited" if overall >= 0.45 else "poor")
    )
    return {
        "overall_score": float(overall),
        "status": status,
        "limitations": sorted(set(limitations)),
    }
