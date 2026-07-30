#!/usr/bin/env python3
"""Run ADRE-CXR reports for a batch of IU X-ray studies."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _select_image(data_root: Path, projections, uid: str) -> Path | None:
    rows = projections[projections["uid"].astype(str) == str(uid)]
    if rows.empty:
        return None

    frontal = rows[rows["projection"].astype(str).str.lower() == "frontal"]
    row = frontal.iloc[0] if not frontal.empty else rows.iloc[0]
    img_name = row["filename"]

    candidates = [
        data_root / "images" / "images_normalized" / img_name,
        data_root / "images" / img_name,
    ]
    return next((path for path in candidates if path.exists()), None)


def _select_evidence(evidence_dir: Path, uid: str) -> Path:
    """Support legacy flat packets and nested evidence-v2 study directories."""
    candidates = [
        evidence_dir / f"{uid}.json",
        evidence_dir / uid / "evidence.json",
        evidence_dir / uid / "evidence_v1_compatible.json",
    ]
    return next((path for path in candidates if path.exists()), candidates[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ADRE-CXR for N IU X-ray studies")
    parser.add_argument("--data-root", default=None, help="Path to IU X-ray dataset")
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--limit", type=int, default=26)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--evidence-dir", default=None)
    parser.add_argument("--index-dir", default=None)
    parser.add_argument("--output-dir", default="outputs/batch_26")
    parser.add_argument("--no-rag", action="store_true")
    parser.add_argument("--no-agentic", action="store_true")
    parser.add_argument("--skip-neural", action="store_true")
    parser.add_argument(
        "--allow-missing-evidence",
        action="store_true",
        help="Run with CPU image features if a precomputed evidence JSON is missing",
    )
    args = parser.parse_args()

    import pandas as pd
    import yaml
    from tqdm import tqdm

    with open(ROOT / args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_root = _project_path(args.data_root or cfg["paths"]["data_root"])
    processed_dir = ROOT / cfg["paths"]["processed_dir"]
    evidence_dir = _project_path(args.evidence_dir or cfg["paths"]["evidence_dir"])
    index_dir = args.index_dir or cfg["paths"]["index_dir"]
    output_dir = _project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_path = processed_dir / f"{args.split}_split.json"
    records = json.loads(split_path.read_text(encoding="utf-8"))
    selected = records[args.offset : args.offset + args.limit]
    projections = pd.read_csv(data_root / "indiana_projections.csv")

    summary_path = output_dir / "summary.jsonl"
    successes = 0
    failures = 0

    with summary_path.open("w", encoding="utf-8") as summary:
        for rec in tqdm(selected, desc="Generating reports"):
            uid = str(rec["uid"])
            image_path = _select_image(data_root, projections, uid)
            evidence_path = _select_evidence(evidence_dir, uid)
            report_path = output_dir / f"{uid}.txt"

            result = {
                "uid": uid,
                "image": str(image_path) if image_path else None,
                "evidence_json": str(evidence_path) if evidence_path.exists() else None,
                "report": str(report_path),
                "status": "pending",
                "error": "",
            }

            if image_path is None:
                result["status"] = "failed"
                result["error"] = "image_not_found"
                failures += 1
                summary.write(json.dumps(result) + "\n")
                continue

            if not evidence_path.exists() and not args.allow_missing_evidence:
                result["status"] = "failed"
                result["error"] = "evidence_json_not_found"
                failures += 1
                summary.write(json.dumps(result) + "\n")
                continue

            cmd = [
                sys.executable,
                str(ROOT / "scripts" / "run_pipeline.py"),
                "--study-id",
                uid,
                "--image",
                str(image_path),
                "--indication",
                rec.get("indication", ""),
                "--index-dir",
                index_dir,
                "--output",
                str(report_path),
            ]
            if evidence_path.exists():
                cmd.extend(["--evidence-json", str(evidence_path)])
            if args.no_rag:
                cmd.append("--no-rag")
            if args.no_agentic:
                cmd.append("--no-agentic")
            if args.skip_neural or not evidence_path.exists():
                cmd.append("--skip-neural")

            completed = subprocess.run(
                cmd,
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            if completed.returncode == 0:
                result["status"] = "ok"
                successes += 1
            else:
                result["status"] = "failed"
                result["error"] = completed.stderr.strip() or completed.stdout.strip()
                failures += 1

            summary.write(json.dumps(result) + "\n")

    print(
        json.dumps(
            {
                "requested": len(selected),
                "successes": successes,
                "failures": failures,
                "summary": str(summary_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
