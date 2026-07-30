"""End-to-end modular evidence-v2 study extractor."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.evidence.anatomy import (
    approximate_anatomy,
    save_anatomy_overlay,
    save_masks,
)
from src.evidence.calibration import ScoreCalibrator
from src.evidence.classifiers import EvidenceModel
from src.evidence.config import config_hash, set_deterministic_seed
from src.evidence.consistency import run_consistency_checks
from src.evidence.fusion import (
    fuse_model_predictions,
    fuse_views,
    generation_guidance,
)
from src.evidence.image_validator import UnreadableImageError, validate_image
from src.evidence.localization import assign_anatomical_label, regions_from_heatmap
from src.evidence.measurements import cardiothoracic_ratio, lung_asymmetry
from src.evidence.preprocessing import preprocess_image
from src.evidence.quality import aggregate_study_quality
from src.evidence.region_analysis import (
    analyze_fixed_grid,
    analyze_lung_regions,
    compact_visual_embedding,
)
from src.evidence.serializer import (
    save_evidence_document,
    save_visual_summary,
    v2_to_legacy_dict,
)
from src.evidence.study_loader import Study
from src.evidence.view_manager import apply_view_policy, manage_views, view_availability

LOGGER = logging.getLogger(__name__)


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {"python": platform.python_version()}
    for package in ("numpy", "Pillow", "opencv-python-headless", "torch", "torchxrayvision"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not_installed"
    return versions


def _cache_key(
    checksums: dict[str, str],
    model_versions: list[str],
    configuration_hash: str,
    extractor_version: str,
) -> str:
    material = json.dumps(
        {
            "checksums": checksums,
            "models": model_versions,
            "config_hash": configuration_hash,
            "extractor_version": extractor_version,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _relative(path: Path, base: Path) -> str:
    return path.relative_to(base).as_posix()


class EvidenceExtractorV2:
    """Process complete studies while isolating failures to individual stages/views."""

    def __init__(
        self,
        config: dict[str, Any],
        models: list[tuple[EvidenceModel, float]] | None = None,
    ):
        self.config = config
        self.models = models or []
        self.configuration_hash = config_hash(config)
        self.calibrator = ScoreCalibrator(config["calibration"])
        set_deterministic_seed(int(config["random_seed"]))

    def extract(
        self,
        study: Study,
        output_root: str | Path,
        force: bool = False,
    ) -> dict[str, Any]:
        study_dir = Path(output_root) / str(study.uid)
        evidence_path = study_dir / "evidence.json"
        study_dir.mkdir(parents=True, exist_ok=True)
        errors: list[dict[str, str]] = []
        validations: dict[str, dict[str, Any]] = {}
        checksums: dict[str, str] = {}
        decoded: dict[str, np.ndarray] = {}

        for source in study.images:
            try:
                validated = validate_image(source.path, self.config["validation"])
                validations[source.view_id] = validated.validation
                checksums[source.view_id] = validated.checksum
                decoded[source.view_id] = validated.image
            except UnreadableImageError as exc:
                validations[source.view_id] = {
                    "readable": False,
                    "quality_status": "poor",
                    "quality_score": 0.0,
                    "warnings": ["unreadable_image"],
                }
                errors.append(
                    {
                        "stage": f"{source.view_id}:validation",
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }
                )

        model_versions = [f"{model.name}:{model.version}" for model, _ in self.models]
        cache_key = _cache_key(
            checksums,
            model_versions,
            self.configuration_hash,
            self.config["extractor_version"],
        )
        if (
            not force
            and self.config["cache"].get("enabled", True)
            and evidence_path.exists()
        ):
            cached = json.loads(evidence_path.read_text(encoding="utf-8"))
            if cached.get("pipeline", {}).get("cache_key") == cache_key:
                cached["_cache_hit"] = True
                return cached

        managed = manage_views(study.images, validations, checksums)
        managed = apply_view_policy(
            managed, bool(self.config["views"].get("use_all_available", True))
        )
        availability = view_availability(managed)
        view_documents: list[dict[str, Any]] = []
        audit_images: dict[str, np.ndarray] = {}
        measurement_candidates: list[tuple[str, dict[str, Any], dict[str, Any]]] = []

        for managed_view in managed:
            source = managed_view.study_image
            view_doc: dict[str, Any] = {
                "view_id": source.view_id,
                "view_type": source.view_type,
                "filename": source.filename,
                "source_path": str(source.path),
                "original_projection": source.original_projection,
                "projection_confidence": source.projection_confidence,
                "projection_basis": source.projection_basis,
                "role": managed_view.role,
                "selected": managed_view.selected,
                "duplicate_of": managed_view.duplicate_of,
                "image_checksum": checksums.get(source.view_id),
                "validation": validations.get(source.view_id, {}),
                "preprocessing": {},
                "global_predictions": [],
                "localizations": [],
                "anatomy": {},
                "region_analysis": [],
                "legacy_fixed_regions": [],
                "embeddings": {},
            }
            if source.view_id not in decoded or not managed_view.selected:
                view_documents.append(view_doc)
                continue
            if (
                validations[source.view_id].get("blank_image")
                and self.config["validation"].get("reject_blank_images", True)
            ):
                errors.append(
                    {
                        "stage": f"{source.view_id}:preprocessing",
                        "type": "InsufficientEvidenceError",
                        "message": "Blank/near-blank image was not classified.",
                    }
                )
                view_documents.append(view_doc)
                continue

            try:
                prepared = preprocess_image(
                    decoded[source.view_id],
                    self.config["preprocessing"],
                    possible_inversion=bool(
                        validations[source.view_id].get("possible_inversion")
                    ),
                )
                normalized_path = study_dir / f"{source.view_id}_normalized.png"
                Image.fromarray(prepared.normalized_original).save(normalized_path)
                view_doc["preprocessing"] = {
                    **prepared.transform,
                    "normalized_original_path": _relative(normalized_path, study_dir),
                }
                if prepared.clahe_enhanced is not None:
                    clahe_path = study_dir / f"{source.view_id}_clahe.png"
                    Image.fromarray(prepared.clahe_enhanced).save(clahe_path)
                    view_doc["preprocessing"]["clahe_path"] = _relative(
                        clahe_path, study_dir
                    )
                audit_images[source.view_id] = prepared.normalized_original
            except Exception as exc:  # noqa: BLE001 - isolate failures by stage
                errors.append(
                    {
                        "stage": f"{source.view_id}:preprocessing",
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }
                )
                view_documents.append(view_doc)
                continue

            if self.config["anatomy"].get("enabled", True):
                anatomy_meta, masks = approximate_anatomy(
                    prepared.normalized_original,
                    source.view_type,
                    self.config["anatomy"],
                )
            else:
                empty_mask = np.zeros_like(prepared.normalized_original, dtype=bool)
                anatomy_meta = {
                    "method": "disabled_for_ablation",
                    "segmentation_quality": 0.0,
                    "limitations": ["Anatomy stage disabled by configuration."],
                }
                masks = {
                    "left_lung": empty_mask.copy(),
                    "right_lung": empty_mask.copy(),
                    "heart": empty_mask.copy(),
                    "mediastinum": empty_mask.copy(),
                }
            mask_paths = save_masks(masks, study_dir / "anatomy" / source.view_id)
            anatomy_overlay_path = (
                study_dir / "anatomy" / f"{source.view_id}_overlay.png"
            )
            save_anatomy_overlay(
                prepared.normalized_original, masks, anatomy_overlay_path
            )
            view_doc["anatomy"] = {
                **anatomy_meta,
                **{
                    key: _relative(Path(path), study_dir)
                    for key, path in mask_paths.items()
                },
                "overlay_path": _relative(anatomy_overlay_path, study_dir),
            }
            view_doc["region_analysis"] = (
                analyze_lung_regions(prepared.normalized_original, masks)
                if self.config["region_analysis"].get("enabled", True)
                else []
            )
            legacy_grid_config = self.config["legacy_fixed_crops"]
            view_doc["legacy_fixed_regions"] = (
                analyze_fixed_grid(
                    prepared.normalized_original,
                    rows=int(legacy_grid_config["rows"]),
                    cols=int(legacy_grid_config["cols"]),
                )
                if legacy_grid_config.get("enabled", True)
                else []
            )
            if (
                view_doc["legacy_fixed_regions"]
                and self.config["output"].get("save_region_crops", True)
            ):
                legacy_crop_dir = study_dir / "crops" / "legacy_fixed"
                legacy_crop_dir.mkdir(parents=True, exist_ok=True)
                for region in view_doc["legacy_fixed_regions"]:
                    box = region["bbox_normalized"]
                    height, width = prepared.normalized_original.shape
                    x0, y0, x1, y1 = (
                        round(box[0] * width),
                        round(box[1] * height),
                        round(box[2] * width),
                        round(box[3] * height),
                    )
                    crop_path = (
                        legacy_crop_dir
                        / f"{source.view_id}_{region['region']}.png"
                    )
                    Image.fromarray(
                        prepared.normalized_original[y0:y1, x0:x1]
                    ).save(crop_path)
                    region["crop_path"] = _relative(crop_path, study_dir)
            if self.config["embeddings"].get("enabled", True) and self.config[
                "embeddings"
            ].get("save_regions", True):
                regional_embedding_dir = study_dir / "embeddings" / "regions"
                regional_embedding_dir.mkdir(parents=True, exist_ok=True)
                for region in view_doc["region_analysis"]:
                    region_name = region["region"]
                    side, zone, _ = region_name.split("_", 2)
                    lung_mask = masks[f"{side}_lung"].copy()
                    ys, _ = np.where(lung_mask)
                    if not len(ys):
                        continue
                    boundaries = np.linspace(int(ys.min()), int(ys.max() + 1), 4).astype(int)
                    zone_index = {"upper": 0, "middle": 1, "lower": 2}[zone]
                    lung_mask[: boundaries[zone_index], :] = False
                    lung_mask[boundaries[zone_index + 1] :, :] = False
                    region_image = np.zeros_like(prepared.normalized_original)
                    region_image[lung_mask] = prepared.normalized_original[lung_mask]
                    embedding = compact_visual_embedding(
                        region_image, self.config["embeddings"]["fallback_size"]
                    )
                    embedding_path = (
                        regional_embedding_dir
                        / f"{source.view_id}_{region_name}.npy"
                    )
                    np.save(embedding_path, embedding)
                    region["embedding_path"] = _relative(embedding_path, study_dir)
                    region["embedding_type"] = "normalized_downsample_not_semantic"
            measurement_candidates.append((source.view_type, anatomy_meta, {
                "ctr": cardiothoracic_ratio(
                    masks, source.view_type, anatomy_meta["method"]
                ),
                "lung_asymmetry": lung_asymmetry(
                    prepared.normalized_original, masks
                ),
            }))

            model_results: list[tuple[dict[str, float], str, str, float]] = []
            learned_embeddings: list[tuple[str, np.ndarray]] = []
            for model, weight in self.models:
                branches = [
                    (
                        "normalized_original",
                        prepared.normalized_original,
                        float(
                            self.config["preprocessing"].get(
                                "original_classification_weight", 1.0
                            )
                        ),
                    )
                ]
                if prepared.clahe_enhanced is not None:
                    branches.append(
                        (
                            "clahe",
                            prepared.clahe_enhanced,
                            float(
                                self.config["preprocessing"]["clahe"].get(
                                    "classification_weight", 0.5
                                )
                            ),
                        )
                    )
                for branch_name, branch_image, branch_weight in branches:
                    try:
                        prediction = model.predict(branch_image, source.view_type)
                        source_name = f"{prediction.model_name}:{branch_name}"
                        model_results.append(
                            (
                                prediction.labels,
                                source_name,
                                prediction.model_version,
                                weight * branch_weight,
                            )
                        )
                        if prediction.embedding is not None:
                            learned_embeddings.append(
                                (
                                    f"{prediction.model_name}_{branch_name}",
                                    prediction.embedding,
                                )
                            )
                    except Exception as exc:  # noqa: BLE001 - isolate model plugins
                        errors.append(
                            {
                                "stage": (
                                    f"{source.view_id}:classification:"
                                    f"{model.name}:{branch_name}"
                                ),
                                "type": type(exc).__name__,
                                "message": str(exc),
                            }
                        )

            assessable = validations[source.view_id]["quality_status"] != "poor"
            view_doc["global_predictions"] = fuse_model_predictions(
                model_results,
                self.calibrator,
                self.config["thresholds"],
                self.config["threshold_defaults"],
                assessable=assessable,
            )

            if self.config["embeddings"].get("enabled", True):
                embedding_dir = study_dir / "embeddings"
                embedding_dir.mkdir(parents=True, exist_ok=True)
                for model_name, embedding in learned_embeddings:
                    embedding_path = (
                        embedding_dir / f"{source.view_id}_{model_name}_global.npy"
                    )
                    np.save(embedding_path, embedding)
                    view_doc["embeddings"][f"{model_name}_global_path"] = _relative(
                        embedding_path, study_dir
                    )
                if not learned_embeddings:
                    fallback = compact_visual_embedding(
                        prepared.normalized_original,
                        self.config["embeddings"]["fallback_size"],
                    )
                    fallback_path = embedding_dir / f"{source.view_id}_visual_fallback.npy"
                    np.save(fallback_path, fallback)
                    view_doc["embeddings"]["global_visual_fallback_path"] = _relative(
                        fallback_path, study_dir
                    )
                    view_doc["embeddings"]["fallback_type"] = (
                        "normalized_downsample_not_semantic"
                    )

            localization_config = self.config["localization"]
            if localization_config.get("enabled", True) and self.models:
                for prediction in view_doc["global_predictions"]:
                    if (
                        float(prediction["interpreted_score"])
                        < float(self.config["classification"]["minimum_analysis_score"])
                    ):
                        continue
                    heatmap = None
                    source_model = None
                    for model, _ in self.models:
                        try:
                            heatmap = model.attribution(
                                prepared.normalized_original, prediction["label"]
                            )
                        except Exception as exc:  # noqa: BLE001 - optional attribution
                            errors.append(
                                {
                                    "stage": f"{source.view_id}:localization:{model.name}",
                                    "type": type(exc).__name__,
                                    "message": str(exc),
                                }
                            )
                        if heatmap is not None:
                            source_model = model.name
                            break
                    if heatmap is None:
                        continue
                    regions = regions_from_heatmap(
                        heatmap,
                        prepared.transform,
                        float(localization_config["heatmap_threshold"]),
                        float(localization_config["min_region_area_fraction"]),
                        int(localization_config["max_regions_per_finding"]),
                        float(localization_config["merge_iou_threshold"]),
                    )
                    for region in regions:
                        region.update(
                            assign_anatomical_label(
                                region["bbox_normalized"],
                                masks["left_lung"],
                                masks["right_lung"],
                            )
                        )
                    heatmap_dir = study_dir / "gradcam"
                    heatmap_dir.mkdir(parents=True, exist_ok=True)
                    safe_label = prediction["label"].lower().replace(" ", "_").replace("/", "_")
                    heatmap_path = heatmap_dir / f"{source.view_id}_{safe_label}.png"
                    Image.fromarray((np.clip(heatmap, 0, 1) * 255).astype(np.uint8)).save(
                        heatmap_path
                    )
                    overlay_array = np.stack([prepared.normalized_original] * 3, axis=-1)
                    overlay_float = overlay_array.astype(np.float32)
                    overlay_float[..., 0] = np.maximum(
                        overlay_float[..., 0], heatmap * 255.0
                    )
                    overlay_float[..., 1] *= 1.0 - 0.45 * heatmap
                    overlay_float[..., 2] *= 1.0 - 0.45 * heatmap
                    heatmap_overlay_path = (
                        heatmap_dir / f"{source.view_id}_{safe_label}_overlay.png"
                    )
                    Image.fromarray(
                        np.clip(overlay_float, 0, 255).astype(np.uint8)
                    ).save(heatmap_overlay_path)
                    crop_paths: list[str] = []
                    if self.config["output"].get("save_region_crops", True):
                        crop_dir = study_dir / "crops"
                        crop_dir.mkdir(parents=True, exist_ok=True)
                        for index, region in enumerate(regions):
                            box = region["bbox_normalized"]
                            height, width = prepared.normalized_original.shape
                            x0, y0, x1, y1 = (
                                int(box[0] * width),
                                int(box[1] * height),
                                max(int(box[0] * width) + 1, int(box[2] * width)),
                                max(int(box[1] * height) + 1, int(box[3] * height)),
                            )
                            crop_path = (
                                crop_dir
                                / f"{source.view_id}_{safe_label}_{index}.png"
                            )
                            Image.fromarray(
                                prepared.normalized_original[y0:y1, x0:x1]
                            ).save(crop_path)
                            region["crop_path"] = _relative(crop_path, study_dir)
                            if self.config["embeddings"].get("save_regions", True):
                                crop_embedding = compact_visual_embedding(
                                    prepared.normalized_original[y0:y1, x0:x1],
                                    self.config["embeddings"]["fallback_size"],
                                )
                                crop_embedding_path = (
                                    study_dir
                                    / "embeddings"
                                    / "regions"
                                    / f"{source.view_id}_{safe_label}_{index}.npy"
                                )
                                crop_embedding_path.parent.mkdir(
                                    parents=True, exist_ok=True
                                )
                                np.save(crop_embedding_path, crop_embedding)
                                region["embedding_path"] = _relative(
                                    crop_embedding_path, study_dir
                                )
                                region["embedding_type"] = (
                                    "normalized_downsample_not_semantic"
                                )
                            crop_paths.append(region["crop_path"])
                    view_doc["localizations"].append(
                        {
                            "finding": prediction["label"],
                            "view_id": source.view_id,
                            "localization_method": localization_config["method"],
                            "interpretation": "model_attended_region",
                            "source_model": source_model,
                            "heatmap_path": _relative(heatmap_path, study_dir),
                            "heatmap_overlay_path": _relative(
                                heatmap_overlay_path, study_dir
                            ),
                            "regions": regions,
                        }
                    )
            view_documents.append(view_doc)

        fusion_config = {
            **self.config["fusion"],
            "view_weights": self.config["views"]["view_weights"],
            "reliability": self.config["reliability"],
        }
        groups, cross_view = fuse_views(
            view_documents,
            fusion_config,
            self.config["thresholds"],
            self.config["threshold_defaults"],
        )
        study_quality = aggregate_study_quality(view_documents, availability)
        checks = run_consistency_checks(
            view_documents,
            groups,
            availability,
            model_disagreement_threshold=float(
                self.config["fusion"]["model_disagreement_threshold"]
            ),
        )
        if not self.models:
            errors.append(
                {
                    "stage": "classification",
                    "type": "ModelUnavailable",
                    "message": "No disease classifier was enabled; only visual/quality evidence was produced.",
                }
            )
        selected_readable = any(
            view.get("selected") and view.get("validation", {}).get("readable")
            for view in view_documents
        )
        status = (
            "failed"
            if not selected_readable
            else "partial_failure"
            if errors
            else "success"
        )
        measurements = {
            "cardiothoracic_ratio": next(
                (
                    item["ctr"]
                    for view_type, _, item in measurement_candidates
                    if view_type == "frontal_pa"
                ),
                {
                    "value": None,
                    "method": "not_computed",
                    "assessable": False,
                    "confidence": 0.0,
                    "view_requirement_met": False,
                    "interpretation": "not_assessable",
                    "limitations": ["No PA frontal view was available."],
                },
            ),
            "per_view_lung_asymmetry": {
                view_type: item["lung_asymmetry"]
                for view_type, _, item in measurement_candidates
            },
            "devices": {
                "assessable": False,
                "method": "no_device_detector_configured",
                "limitations": [
                    "General disease-classifier scores are not used for device positioning."
                ],
            },
        }
        document: dict[str, Any] = {
            "schema_version": str(self.config["schema_version"]),
            "uid": str(study.uid),
            "processing_status": status,
            "study_metadata": {
                "dataset": "IU_XRAY",
                "available_views": [
                    view["view_type"]
                    for view in view_documents
                    if view.get("selected", True)
                ],
                "processing_timestamp": datetime.now(timezone.utc).isoformat(),
                "patient_metadata": study.metadata,
            },
            "pipeline": {
                "extractor_version": self.config["extractor_version"],
                "config_hash": self.configuration_hash,
                "cache_key": cache_key,
                "random_seed": int(self.config["random_seed"]),
                "models": [
                    {
                        "name": model.name,
                        "version": model.version,
                        "ensemble_weight": weight,
                    }
                    for model, weight in self.models
                ],
                "calibration": self.calibrator.metadata(),
                "package_versions": _package_versions(),
                "code_version": os.getenv("ADRE_CODE_VERSION", "working_tree"),
            },
            "views": view_documents,
            "view_availability": availability,
            "fused_findings": groups,
            "measurements": measurements,
            "cross_view_analysis": cross_view,
            "consistency_checks": checks,
            "study_quality": study_quality,
            "generation_guidance": generation_guidance(groups),
            "errors": errors,
        }
        save_evidence_document(
            document,
            evidence_path,
            pretty=bool(self.config["output"].get("pretty_json", True)),
        )
        if self.config["output"].get("save_legacy_packet", True):
            legacy_path = study_dir / "evidence_v1_compatible.json"
            legacy_path.write_text(
                json.dumps(v2_to_legacy_dict(document), indent=2, allow_nan=False),
                encoding="utf-8",
            )
        if self.config["output"].get("save_visualizations", True):
            save_visual_summary(
                document, audit_images, study_dir / "prediction_summary.png"
            )
        document["_cache_hit"] = False
        return document
