"""IU-Xray study loading with projection classification and full view retention."""

from __future__ import annotations

import csv
import hashlib
import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.evidence.schemas import ViewType

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class StudyImage:
    view_id: str
    filename: str
    path: Path
    view_type: str
    original_projection: str
    projection_confidence: float
    projection_basis: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Study:
    uid: str
    images: list[StudyImage]
    metadata: dict[str, Any] = field(default_factory=dict)


def classify_projection(
    projection: str | None,
    filename: str = "",
    metadata: dict[str, Any] | None = None,
) -> tuple[str, float, list[str]]:
    """Map noisy IU projection text to the v2 controlled vocabulary."""
    metadata = metadata or {}
    text = " ".join(
        [
            str(projection or ""),
            filename,
            str(metadata.get("view", "")),
            str(metadata.get("projection", "")),
        ]
    ).lower()
    basis: list[str] = []
    if "lateral" in text or "_lat" in text or "-lat" in text:
        basis.append("projection_or_filename_contains_lateral")
        return ViewType.LATERAL.value, 0.98, basis
    if "portable" in text or " ap " in f" {text} " or text.startswith("ap"):
        basis.append("projection_or_filename_contains_ap")
        return ViewType.FRONTAL_AP.value, 0.92, basis
    if " pa " in f" {text} " or text.startswith("pa"):
        basis.append("projection_or_filename_contains_pa")
        return ViewType.FRONTAL_PA.value, 0.95, basis
    if "frontal" in text:
        # IU labels many frontal studies without differentiating PA/AP. Most
        # standard outpatient IU frontal images are PA, but retain lower
        # confidence so downstream logic does not over-trust the assumption.
        basis.append("iu_frontal_default_pa")
        return ViewType.FRONTAL_PA.value, 0.65, basis
    if any(token in text for token in ("oblique", "decubitus", "lordotic")):
        basis.append("nonstandard_projection_string")
        return ViewType.OTHER.value, 0.90, basis
    return ViewType.UNKNOWN.value, 0.0, ["projection_unresolved"]


def file_checksum(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


class IUStudyLoader:
    """Load all image rows for a UID from ``indiana_projections.csv``."""

    def __init__(self, data_root: str | Path):
        self.data_root = Path(data_root)
        self.projections_path = self.data_root / "indiana_projections.csv"
        if not self.projections_path.exists():
            raise FileNotFoundError(f"Missing projections CSV: {self.projections_path}")
        with self.projections_path.open(
            "r", encoding="utf-8-sig", newline=""
        ) as handle:
            self._rows = list(csv.DictReader(handle))

    def available_uids(self) -> set[str]:
        return {
            str(row.get("uid", "")) for row in self._rows if row.get("uid") is not None
        }

    def _resolve_image(self, filename: str) -> Path:
        candidates = [
            self.data_root / "images" / "images_normalized" / filename,
            self.data_root / "images" / filename,
            self.data_root / filename,
        ]
        return next((path for path in candidates if path.exists()), candidates[0])

    def load(self, uid: str | int, metadata: dict[str, Any] | None = None) -> Study:
        uid_text = str(uid)
        selected = [row for row in self._rows if str(row.get("uid", "")) == uid_text]
        images: list[StudyImage] = []
        for index, row in enumerate(selected):
            filename = str(row.get("filename", "")).strip()
            projection = str(row.get("projection", "")).strip()
            view_type, confidence, basis = classify_projection(
                projection, filename, metadata=row
            )
            images.append(
                StudyImage(
                    view_id=f"{uid_text}_{view_type}_{index}",
                    filename=filename,
                    path=self._resolve_image(filename),
                    view_type=view_type,
                    original_projection=projection,
                    projection_confidence=confidence,
                    projection_basis=basis,
                )
            )
        if not images:
            LOGGER.warning("No projection rows found for UID %s", uid_text)
        return Study(uid=uid_text, images=images, metadata=metadata or {})

    def iter_studies(self, records: Iterable[dict[str, Any]]) -> Iterable[Study]:
        for record in records:
            yield self.load(record["uid"], metadata=record)
