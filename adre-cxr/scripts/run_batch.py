#!/usr/bin/env python3
"""Batch runner for ADRE-CXR pipeline across 10 records."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw" / "chest-xrays-indiana-university"
IMAGES_DIR = DATA_DIR / "images" / "images_normalized"
PROJECTIONS_CSV = DATA_DIR / "indiana_projections.csv"
REPORTS_CSV = DATA_DIR / "indiana_reports.csv"
OUTPUT_DIR = ROOT / "batch_output"


def load_reports() -> dict[str, str]:
    """Load study_id -> indication mapping."""
    reports = {}
    if not REPORTS_CSV.exists():
        print(f"Error: reports CSV not found at {REPORTS_CSV}")
        return reports

    with open(REPORTS_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            uid = row.get("uid")
            indication = row.get("indication", "").strip()
            if uid:
                reports[uid] = indication if indication else "eval abnormality"
    return reports


def find_records(limit: int = 10) -> list[dict[str, str]]:
    """Find frontal projection images that exist in images_normalized."""
    reports_map = load_reports()
    records = []

    if not PROJECTIONS_CSV.exists():
        print(f"Error: projections CSV not found at {PROJECTIONS_CSV}")
        return records

    with open(PROJECTIONS_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if len(records) >= limit:
                break
            uid = row.get("uid")
            filename = row.get("filename")
            projection = row.get("projection")

            if projection == "Frontal" and uid and filename:
                img_path = IMAGES_DIR / filename
                if img_path.exists():
                    records.append({
                        "study_id": uid,
                        "image_path": str(img_path),
                        "indication": reports_map.get(uid, "eval chest radiograph")
                    })
    return records


def main():
    parser = argparse.ArgumentParser(description="Batch execute ADRE-CXR pipeline.")
    parser.add_argument("--limit", type=int, default=10, help="Number of records to process.")
    parser.add_argument("--no-agentic", action="store_true", help="Skip agent planning loop for speed.")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)
    records = find_records(args.limit)

    if not records:
        print("No valid frontal images found.")
        sys.exit(1)

    print(f"Found {len(records)} valid records. Starting batch run...")
    results = []

    python_executable = sys.executable

    for idx, rec in enumerate(records):
        print("\n" + "=" * 50)
        print(f"Processing Record {idx+1}/{len(records)}: Study ID {rec['study_id']}")
        print(f"Image: {rec['image_path']}")
        print(f"Indication: {rec['indication']}")
        print("=" * 50)

        cmd = [
            python_executable,
            "scripts/run_pipeline.py",
            "--study-id", rec["study_id"],
            "--image", rec["image_path"],
            "--indication", rec["indication"],
            "--output", str(OUTPUT_DIR / f"report_{rec['study_id']}.txt")
        ]
        if args.no_agentic:
            cmd.append("--no-agentic")

        try:
            res = subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=str(ROOT))
            print(f"Record {rec['study_id']} completed successfully.")
            
            # Extract output JSON block
            stdout = res.stdout.strip()
            # Find the starting '{' of the print JSON output
            json_start = stdout.find("{")
            if json_start != -1:
                structured_data = json.loads(stdout[json_start:])
                print(json.dumps(structured_data, indent=2))
                rec["status"] = "success"
                rec["structured"] = structured_data.get("structured")
            else:
                rec["status"] = "success"
                rec["raw_stdout"] = stdout
        except subprocess.CalledProcessError as e:
            print(f"Error processing record {rec['study_id']}:")
            print(e.stderr)
            rec["status"] = "failed"
            rec["error"] = e.stderr
        
        results.append(rec)

    # Save summary report
    summary_path = OUTPUT_DIR / "batch_summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nBatch processing finished. Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
