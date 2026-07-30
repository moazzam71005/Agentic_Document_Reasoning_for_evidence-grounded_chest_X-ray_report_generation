from src.evidence.calibration import ScoreCalibrator, interpret_score

DEFAULTS = {
    "absent_max": 0.2,
    "probably_absent_max": 0.35,
    "uncertain_max": 0.5,
    "present_min": 0.7,
}


def test_uncalibrated_scores_are_explicitly_heuristic():
    calibrator = ScoreCalibrator({"method": "none", "calibrated": False})
    assert calibrator.apply("Effusion", 0.61) == 0.61
    assert calibrator.metadata()["confidence_type"] == "heuristic_model_score"


def test_all_status_bands():
    assert interpret_score("x", 0.1, {}, DEFAULTS) == "absent"
    assert interpret_score("x", 0.3, {}, DEFAULTS) == "probably_absent"
    assert interpret_score("x", 0.4, {}, DEFAULTS) == "uncertain"
    assert interpret_score("x", 0.6, {}, DEFAULTS) == "probably_present"
    assert interpret_score("x", 0.8, {}, DEFAULTS) == "present"
    assert interpret_score("x", 0.8, {}, DEFAULTS, assessable=False) == "not_assessable"
