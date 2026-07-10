"""Tool planning — request additional views/analyses."""

from __future__ import annotations

from typing import Any


def plan_tools(gaps: list[dict], views_used: set[str]) -> list[dict[str, Any]]:
    """Map gaps to view/tool requests."""
    requests = []
    for gap in gaps:
        if gap["id"] == "findings_lungs" and "zone_upper_left" not in views_used:
            requests.append({"tool": "zone_crop", "view_id": "zone_upper_left"})
            requests.append({"tool": "zone_crop", "view_id": "zone_lower_right"})
        elif gap["id"] == "findings_quality" and "clahe" not in views_used:
            requests.append({"tool": "clahe", "view_id": "clahe"})
        elif gap["id"] == "findings_heart" and "original" not in views_used:
            requests.append({"tool": "load", "view_id": "original"})
    return requests[:3]  # cap per iteration
