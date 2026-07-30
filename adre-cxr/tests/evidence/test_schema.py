from copy import deepcopy

from src.evidence.schemas import validate_evidence_document


def valid_document():
    return {
        "schema_version": "2.0",
        "uid": "1",
        "processing_status": "success",
        "study_metadata": {},
        "pipeline": {},
        "views": [
            {
                "view_type": "frontal_pa",
                "global_predictions": [
                    {
                        "status": "uncertain",
                        "raw_score": 0.4,
                        "interpreted_score": 0.4,
                    }
                ],
                "localizations": [
                    {"regions": [{"bbox_normalized": [0.1, 0.2, 0.8, 0.9]}]}
                ],
            }
        ],
        "view_availability": {},
        "fused_findings": {
            "positive": [],
            "negative": [],
            "uncertain": [],
            "not_assessable": [],
        },
        "consistency_checks": [],
        "study_quality": {},
        "generation_guidance": {},
        "errors": [],
    }


def test_valid_schema_document():
    assert validate_evidence_document(valid_document()) == []


def test_nan_and_bad_bbox_are_rejected():
    document = deepcopy(valid_document())
    document["views"][0]["global_predictions"][0]["raw_score"] = float("nan")
    document["views"][0]["localizations"][0]["regions"][0]["bbox_normalized"] = [
        -0.1,
        0,
        1,
        2,
    ]
    errors = validate_evidence_document(document)
    assert any("NaN" in error for error in errors)
    assert any("coordinates" in error for error in errors)
