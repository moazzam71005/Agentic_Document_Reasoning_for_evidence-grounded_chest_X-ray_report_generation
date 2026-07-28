"""Provenance-tagged evidence packet assembly (Φ vector)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.evidence.image_features import extract_features
from src.evidence.quality_indicators import assess_quality
from src.evidence.view_builder import ImageView, build_views


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


@dataclass
class Provenance:
    view_id: str
    zone: str | None
    params: dict[str, Any]
    tool: str

    @classmethod
    def from_view(cls, view: ImageView) -> "Provenance":
        zone = view.params.get("row")
        if zone and view.params.get("col"):
            zone = f"{view.params['row']}_{view.params['col']}"
        return cls(view_id=view.view_id, zone=zone, params=view.params, tool=view.tool)


@dataclass
class EvidencePacket:
    study_id: str
    image_path: str
    patient_metadata: dict[str, Any] = field(default_factory=dict)
    global_pathology_probs: dict[str, float] = field(default_factory=dict)
    global_abnormality_score: float = 0.0
    regional_burden: dict[str, Any] = field(default_factory=dict)
    image_features: dict[str, float] = field(default_factory=dict)
    quality_indicators: dict[str, Any] = field(default_factory=dict)
    provenance: list[dict] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, default=_json_default),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "EvidencePacket":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(**data)


def build_evidence_packet(
    study_id: str,
    image_path: str,
    patient_metadata: dict | None = None,
    neural: Any | None = None,
    rows: int = 3,
    cols: int = 2,
) -> EvidencePacket:
    """
    Build a full evidence packet for one image.
    Pass `neural=None` to skip GPU classifier (features + quality only).
    """
    views = build_views(image_path, rows=rows, cols=cols)
    original = next(v for v in views if v.view_id == "original")

    packet = EvidencePacket(
        study_id=study_id,
        image_path=image_path,
        patient_metadata=patient_metadata or {},
        image_features=extract_features(original.image),
        quality_indicators=assess_quality(original.image),
        provenance=[asdict(Provenance.from_view(v)) for v in views],
    )

    if neural is not None:
        packet.global_pathology_probs = neural.predict(original.image)
        packet.global_abnormality_score = neural.global_abnormality_score(
            packet.global_pathology_probs
        )
        zone_probs = {}
        for v in views:
            if v.view_id.startswith("zone_"):
                zone_probs[v.view_id] = neural.predict(v.image)
        packet.regional_burden = neural.regional_burden(zone_probs)

    return packet
