"""Cross-view consistency checks."""

from __future__ import annotations


def sanity_check(evidence_items: list[dict]) -> list[str]:
    """Flag conflicts between global and regional evidence."""
    warnings = []
    global_score = None
    max_zone = 0.0

    for item in evidence_items:
        if "global_abnormality_score" in item:
            global_score = item["global_abnormality_score"]
        if "regional_burden" in item:
            for zone_data in item["regional_burden"].values():
                if isinstance(zone_data, dict):
                    max_zone = max(max_zone, zone_data.get("max_pathology_prob", 0))

    if global_score is not None and global_score > 0.7 and max_zone < 0.3:
        warnings.append(
            "High global abnormality but low regional burden — re-analyze zones"
        )
    return warnings
