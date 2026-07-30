"""Safe evidence serialization, visual summaries, and v2-to-v1 conversion."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.evidence.schemas import validate_evidence_document


def json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(child) for child in value]
    if isinstance(value, Path):
        return str(value)
    return value


def save_evidence_document(
    document: dict[str, Any], path: str | Path, pretty: bool = True
) -> None:
    safe = json_safe(document)
    errors = validate_evidence_document(safe)
    if errors:
        raise ValueError("Invalid evidence v2 document:\n" + "\n".join(errors))
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            safe,
            indent=2 if pretty else None,
            sort_keys=not pretty,
            allow_nan=False,
        ),
        encoding="utf-8",
    )


def v2_to_legacy_dict(document: dict[str, Any]) -> dict[str, Any]:
    """Map v2 evidence to the exact packet fields consumed by the generator."""
    fused = [
        item
        for group in document.get("fused_findings", {}).values()
        for item in group
    ]
    probabilities = {
        item["name"]: float(item.get("score", 0.0))
        for item in fused
        if item.get("status") != "not_assessable"
    }
    views = [view for view in document.get("views", []) if view.get("selected", True)]
    frontal = next(
        (
            view
            for view in views
            if view.get("view_type") in {"frontal_pa", "frontal_ap"}
        ),
        views[0] if views else {},
    )
    regional_burden: dict[str, Any] = {}
    for view in views:
        for region in view.get("region_analysis", []):
            regional_burden[f"{view['view_id']}:{region['region']}"] = {
                "mean_intensity": region.get("mean_intensity", 0.0),
                "std_intensity": region.get("std_intensity", 0.0),
                "method": region.get("method"),
            }
        for region in view.get("legacy_fixed_regions", []):
            regional_burden[f"{view['view_id']}:legacy:{region['region']}"] = {
                "mean_intensity": region.get("mean_intensity", 0.0),
                "std_intensity": region.get("std_intensity", 0.0),
                "method": region.get("method"),
            }
    return {
        "study_id": str(document["uid"]),
        "image_path": str(frontal.get("source_path", "")),
        "patient_metadata": document.get("study_metadata", {}).get(
            "patient_metadata", {}
        ),
        "global_pathology_probs": probabilities,
        "global_abnormality_score": max(probabilities.values()) if probabilities else 0.0,
        "regional_burden": regional_burden,
        "image_features": {
            "evidence_schema_version": 2.0,
            "study_quality_score": document.get("study_quality", {}).get(
                "overall_score", 0.0
            ),
        },
        "quality_indicators": document.get("study_quality", {}),
        "provenance": [
            {
                "view_id": view.get("view_id"),
                "zone": None,
                "params": {
                    "view_type": view.get("view_type"),
                    "validation": view.get("validation", {}),
                    "generation_guidance": document.get("generation_guidance", {}),
                },
                "tool": "evidence_v2_adapter",
            }
            for view in views
        ],
        "created_at": document.get("study_metadata", {}).get(
            "processing_timestamp", ""
        ),
    }


def save_visual_summary(
    document: dict[str, Any],
    view_images: dict[str, np.ndarray],
    output_path: str | Path,
) -> None:
    """Render a compact, deterministic study audit image."""
    selected = [view for view in document["views"] if view.get("selected", True)]
    if not selected:
        return
    tile_size = 384
    header = 150
    canvas = Image.new(
        "RGB", (tile_size * len(selected), tile_size + header), color="white"
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    positives = document["fused_findings"]["positive"][:5]
    warnings = document["consistency_checks"][:3]
    summary = (
        f"UID {document['uid']} | status={document['processing_status']} | "
        f"quality={document['study_quality']['overall_score']:.2f}\n"
        + "Positive: "
        + (", ".join(f"{x['name']} {x['score']:.2f}" for x in positives) or "none")
        + "\n"
        + "Warnings: "
        + ("; ".join(item["message"] for item in warnings) or "none")
    )
    draw.multiline_text((8, 8), summary, fill="black", font=font, spacing=4)
    for index, view in enumerate(selected):
        array = view_images.get(view["view_id"])
        if array is None:
            continue
        tile = Image.fromarray(array.astype(np.uint8)).convert("RGB")
        tile.thumbnail((tile_size, tile_size))
        x = index * tile_size + (tile_size - tile.width) // 2
        y = header + (tile_size - tile.height) // 2
        canvas.paste(tile, (x, y))
        draw.text(
            (index * tile_size + 8, header + 8),
            f"{view['view_type']} | q={view['validation']['quality_score']:.2f}",
            fill="yellow",
            font=font,
        )
        for localization in view.get("localizations", []):
            for region in localization.get("regions", []):
                box = region["bbox_normalized"]
                x0 = index * tile_size + int(box[0] * tile_size)
                y0 = header + int(box[1] * tile_size)
                x1 = index * tile_size + int(box[2] * tile_size)
                y1 = header + int(box[3] * tile_size)
                draw.rectangle((x0, y0, x1, y1), outline="red", width=2)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
