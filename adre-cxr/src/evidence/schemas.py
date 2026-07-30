"""Schema constants and dependency-free validation for evidence v2."""

from __future__ import annotations

import math
from enum import Enum
from typing import Any


class FindingStatus(str, Enum):
    PRESENT = "present"
    PROBABLY_PRESENT = "probably_present"
    UNCERTAIN = "uncertain"
    PROBABLY_ABSENT = "probably_absent"
    ABSENT = "absent"
    NOT_ASSESSABLE = "not_assessable"


class ViewType(str, Enum):
    FRONTAL_PA = "frontal_pa"
    FRONTAL_AP = "frontal_ap"
    LATERAL = "lateral"
    OTHER = "other"
    UNKNOWN = "unknown"


VALID_FINDING_STATUSES = {status.value for status in FindingStatus}
VALID_VIEW_TYPES = {view.value for view in ViewType}


EVIDENCE_V2_JSON_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "ADRE-CXR Evidence v2",
    "type": "object",
    "required": [
        "schema_version",
        "uid",
        "processing_status",
        "study_metadata",
        "pipeline",
        "views",
        "view_availability",
        "fused_findings",
        "consistency_checks",
        "study_quality",
        "generation_guidance",
        "errors",
    ],
    "properties": {
        "schema_version": {"const": "2.0"},
        "uid": {"type": "string", "minLength": 1},
        "processing_status": {"enum": ["success", "partial_failure", "failed"]},
        "views": {"type": "array"},
        "fused_findings": {
            "type": "object",
            "required": ["positive", "negative", "uncertain", "not_assessable"],
        },
        "consistency_checks": {"type": "array"},
        "errors": {"type": "array"},
    },
}


def _check_finite(value: Any, path: str, errors: list[str]) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        errors.append(f"{path}: NaN and Infinity are not allowed")
    elif isinstance(value, dict):
        for key, child in value.items():
            _check_finite(child, f"{path}.{key}", errors)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _check_finite(child, f"{path}[{index}]", errors)


def validate_evidence_document(document: dict[str, Any]) -> list[str]:
    """Return validation errors; an empty list means the document is valid."""
    errors: list[str] = []
    for field in EVIDENCE_V2_JSON_SCHEMA["required"]:
        if field not in document:
            errors.append(f"$: missing required field '{field}'")
    if document.get("schema_version") != "2.0":
        errors.append("$.schema_version: expected '2.0'")
    if not str(document.get("uid", "")):
        errors.append("$.uid: must be a non-empty string")
    if document.get("processing_status") not in {
        "success",
        "partial_failure",
        "failed",
    }:
        errors.append("$.processing_status: invalid value")

    views = document.get("views", [])
    if not isinstance(views, list):
        errors.append("$.views: must be an array")
        views = []
    for i, view in enumerate(views):
        view_type = view.get("view_type")
        if view_type not in VALID_VIEW_TYPES:
            errors.append(f"$.views[{i}].view_type: invalid value '{view_type}'")
        for prediction in view.get("global_predictions", []):
            status = prediction.get("status")
            if status not in VALID_FINDING_STATUSES:
                errors.append(
                    f"$.views[{i}].global_predictions.status: invalid '{status}'"
                )
            for score_key in ("raw_score", "interpreted_score"):
                score = prediction.get(score_key)
                if score is not None and not 0.0 <= float(score) <= 1.0:
                    errors.append(
                        f"$.views[{i}].global_predictions.{score_key}: outside [0, 1]"
                    )
        for localization in view.get("localizations", []):
            for region in localization.get("regions", []):
                bbox = region.get("bbox_normalized", [])
                if len(bbox) != 4 or any(not 0.0 <= float(x) <= 1.0 for x in bbox):
                    errors.append(
                        f"$.views[{i}].localizations.bbox_normalized: invalid coordinates"
                    )

    groups = document.get("fused_findings", {})
    if not isinstance(groups, dict):
        errors.append("$.fused_findings: must be an object")
    else:
        for group in ("positive", "negative", "uncertain", "not_assessable"):
            if not isinstance(groups.get(group), list):
                errors.append(f"$.fused_findings.{group}: must be an array")

    _check_finite(document, "$", errors)
    return errors
