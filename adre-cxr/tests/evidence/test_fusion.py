import numpy as np

from src.evidence.calibration import ScoreCalibrator
from src.evidence.fusion import fuse_model_predictions, fuse_views

DEFAULTS = {
    "absent_max": 0.2,
    "probably_absent_max": 0.35,
    "uncertain_max": 0.5,
    "present_min": 0.7,
}
CONFIG = {
    "conflict_threshold": 0.35,
    "view_weights": {"frontal_pa": 1.0, "lateral": 0.9},
    "reliability": {
        "model_agreement_weight": 0.30,
        "view_agreement_weight": 0.25,
        "image_quality_weight": 0.15,
        "localization_weight": 0.15,
        "anatomy_weight": 0.15,
    },
}


def _view(view_id, view_type, score, quality=1.0):
    return {
        "view_id": view_id,
        "view_type": view_type,
        "selected": True,
        "validation": {"quality_score": quality},
        "anatomy": {"segmentation_quality": 0.8},
        "localizations": [],
        "global_predictions": [
            {
                "label": "Effusion",
                "interpreted_score": score,
                "status": "present" if score >= 0.7 else "uncertain",
                "model_disagreement": 0.0,
            }
        ],
    }


def test_views_agree():
    groups, cross = fuse_views(
        [_view("f", "frontal_pa", 0.8), _view("l", "lateral", 0.85)],
        CONFIG,
        {},
        DEFAULTS,
    )
    assert groups["positive"][0]["view_agreement"] == "agree"
    assert cross["findings"][0]["fused"]["status"] == "present"


def test_view_conflict_and_missing_view():
    groups, _ = fuse_views(
        [_view("f", "frontal_pa", 0.85), _view("l", "lateral", 0.2)],
        CONFIG,
        {},
        DEFAULTS,
    )
    all_findings = [item for group in groups.values() for item in group]
    assert all_findings[0]["view_agreement"] == "conflicting"

    groups, _ = fuse_views([_view("f", "frontal_pa", 0.8)], CONFIG, {}, DEFAULTS)
    assert groups["positive"][0]["view_agreement"] == "single_view"


def test_low_quality_view_is_downweighted():
    groups, _ = fuse_views(
        [_view("f", "frontal_pa", 0.8), _view("l", "lateral", 0.1, quality=0.05)],
        CONFIG,
        {},
        DEFAULTS,
    )
    assert groups["positive"][0]["score"] > 0.7


def test_nonfinite_model_output_is_ignored():
    calibrator = ScoreCalibrator({"method": "none", "calibrated": False})
    predictions = [
        ({"Effusion": float("nan")}, "bad", "1", 1.0),
        ({"Effusion": 0.8}, "good", "1", 1.0),
    ]
    fused = fuse_model_predictions(
        predictions, calibrator, {}, DEFAULTS, assessable=True
    )
    assert len(fused) == 1
    assert np.isfinite(fused[0]["interpreted_score"])
    assert fused[0]["interpreted_score"] == 0.8
