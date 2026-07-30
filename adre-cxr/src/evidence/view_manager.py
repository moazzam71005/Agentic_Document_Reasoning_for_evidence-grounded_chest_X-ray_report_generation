"""Multi-view selection, duplicate handling, and availability summaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.evidence.study_loader import StudyImage


@dataclass
class ManagedView:
    study_image: StudyImage
    role: str
    selected: bool
    duplicate_of: str | None = None


def manage_views(
    images: list[StudyImage],
    validations: dict[str, dict[str, Any]],
    checksums: dict[str, str],
) -> list[ManagedView]:
    """Keep all unique views and choose best-quality duplicates as primary."""
    duplicate_groups: dict[str, list[StudyImage]] = {}
    for image in images:
        key = checksums.get(image.view_id) or f"missing:{image.view_id}"
        duplicate_groups.setdefault(key, []).append(image)

    unique_images: list[StudyImage] = []
    managed: list[ManagedView] = []
    for candidates in duplicate_groups.values():
        ranked = sorted(
            candidates,
            key=lambda item: (
                float(validations.get(item.view_id, {}).get("quality_score", 0.0)),
                item.projection_confidence,
                item.filename,
            ),
            reverse=True,
        )
        winner = ranked[0]
        unique_images.append(winner)
        for duplicate in ranked[1:]:
            managed.append(
                ManagedView(
                    study_image=duplicate,
                    role="duplicate",
                    selected=False,
                    duplicate_of=winner.view_id,
                )
            )

    by_type: dict[str, list[StudyImage]] = {}
    for image in unique_images:
        by_type.setdefault(image.view_type, []).append(image)

    for candidates in by_type.values():
        ranked = sorted(
            candidates,
            key=lambda item: (
                float(validations.get(item.view_id, {}).get("quality_score", 0.0)),
                item.projection_confidence,
                item.filename,
            ),
            reverse=True,
        )
        for index, image in enumerate(ranked):
            managed.append(
                ManagedView(
                    study_image=image,
                    role="primary" if index == 0 else "secondary",
                    selected=True,
                )
            )
    return sorted(managed, key=lambda item: item.study_image.view_id)


def view_availability(managed: list[ManagedView]) -> dict[str, bool]:
    selected_types = {
        item.study_image.view_type for item in managed if item.selected
    }
    frontal = bool({"frontal_pa", "frontal_ap"} & selected_types)
    lateral = "lateral" in selected_types
    return {
        "frontal": frontal,
        "lateral": lateral,
        "multi_view_complete": frontal and lateral,
        "lateral_only": lateral and not frontal,
    }


def apply_view_policy(
    managed: list[ManagedView], use_all_available: bool
) -> list[ManagedView]:
    """Apply the ablation policy while retaining unselected view metadata."""
    if use_all_available:
        return managed
    candidates = [
        item
        for item in managed
        if item.selected
        and item.study_image.view_type in {"frontal_pa", "frontal_ap"}
    ]
    if not candidates:
        candidates = [item for item in managed if item.selected]
    keep = candidates[0].study_image.view_id if candidates else None
    for item in managed:
        if item.selected and item.study_image.view_id != keep:
            item.selected = False
            if item.role != "duplicate":
                item.role = "ablation_excluded"
    return managed
