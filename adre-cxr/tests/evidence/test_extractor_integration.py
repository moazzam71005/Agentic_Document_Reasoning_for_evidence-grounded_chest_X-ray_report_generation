from pathlib import Path

import numpy as np
from PIL import Image

from src.evidence.config import load_evidence_config
from src.evidence.extractor import EvidenceExtractorV2
from src.evidence.schemas import validate_evidence_document
from src.evidence.serializer import v2_to_legacy_dict
from src.evidence.study_loader import IUStudyLoader


def test_multiview_extraction_cache_and_legacy_adapter(tmp_path: Path):
    data_root = tmp_path / "iu"
    image_dir = data_root / "images" / "images_normalized"
    image_dir.mkdir(parents=True)
    gradient = np.tile(np.arange(256, dtype=np.uint8), (256, 1))
    Image.fromarray(gradient).save(image_dir / "front.png")
    Image.fromarray(np.rot90(gradient)).save(image_dir / "lateral.png")
    (data_root / "indiana_projections.csv").write_text(
        "uid,filename,projection\n11,front.png,PA\n11,lateral.png,Lateral\n",
        encoding="utf-8",
    )

    config_path = Path(__file__).resolve().parents[2] / "configs" / "evidence_v2.yaml"
    config = load_evidence_config(config_path)
    config["output"]["save_visualizations"] = True
    loader = IUStudyLoader(data_root)
    extractor = EvidenceExtractorV2(config, models=[])

    first = extractor.extract(loader.load("11"), tmp_path / "out")
    assert first["processing_status"] == "partial_failure"
    assert len(first["views"]) == 2
    assert first["view_availability"]["multi_view_complete"]
    assert validate_evidence_document(first) == []
    assert (tmp_path / "out" / "11" / "prediction_summary.png").exists()
    assert (tmp_path / "out" / "11" / "evidence_v1_compatible.json").exists()

    legacy = v2_to_legacy_dict(first)
    assert legacy["study_id"] == "11"
    assert legacy["image_features"]["evidence_schema_version"] == 2.0

    second = extractor.extract(loader.load("11"), tmp_path / "out")
    assert second["_cache_hit"] is True
    assert second["pipeline"]["cache_key"] == first["pipeline"]["cache_key"]
