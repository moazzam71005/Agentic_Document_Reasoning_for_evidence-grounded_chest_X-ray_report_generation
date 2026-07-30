from pathlib import Path

from src.evidence.study_loader import IUStudyLoader, StudyImage, classify_projection
from src.evidence.view_manager import manage_views, view_availability


def test_projection_classification():
    assert classify_projection("PA", "x.png")[0] == "frontal_pa"
    assert classify_projection("AP Portable", "x.png")[0] == "frontal_ap"
    assert classify_projection("Lateral", "x.png")[0] == "lateral"
    assert classify_projection("Frontal", "x.png")[0] == "frontal_pa"


def test_study_loader_retains_all_rows(tmp_path: Path):
    (tmp_path / "indiana_projections.csv").write_text(
        "uid,filename,projection\n7,a.png,Frontal\n7,b.png,Lateral\n8,c.png,Frontal\n",
        encoding="utf-8",
    )
    loader = IUStudyLoader(tmp_path)
    study = loader.load("7")
    assert len(study.images) == 2
    assert {image.view_type for image in study.images} == {"frontal_pa", "lateral"}


def test_duplicate_views_are_retained_but_not_selected():
    images = [
        StudyImage("a", "a.png", Path("a.png"), "frontal_pa", "PA", 1.0),
        StudyImage("b", "b.png", Path("b.png"), "frontal_pa", "PA", 1.0),
        StudyImage("c", "c.png", Path("c.png"), "lateral", "Lateral", 1.0),
    ]
    managed = manage_views(
        images,
        {
            "a": {"quality_score": 0.8},
            "b": {"quality_score": 0.9},
            "c": {"quality_score": 0.9},
        },
        {"a": "same", "b": "same", "c": "different"},
    )
    duplicate = next(item for item in managed if item.study_image.view_id == "a")
    assert not duplicate.selected
    assert duplicate.duplicate_of == "b"
    assert view_availability(managed)["multi_view_complete"]
