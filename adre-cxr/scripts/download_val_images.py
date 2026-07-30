#!/usr/bin/env python3
"""Download only the specific chest X-ray images needed for the validation split from HuggingFace."""

from __future__ import annotations

import csv
import sys
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_prep.parse_reports import parse_reports_df

def main():
    reports_csv = ROOT / "data" / "raw" / "chest-xrays-indiana-university" / "indiana_reports.csv"
    projections_csv = ROOT / "data" / "raw" / "chest-xrays-indiana-university" / "indiana_projections.csv"
    local_dir = ROOT / "data" / "raw" / "chest-xrays-indiana-university"

    # 1. Load reports and split to get exact val_records
    df_raw = pd.read_csv(str(reports_csv))
    reports = parse_reports_df(df_raw)
    _, val_records = train_test_split(reports, test_size=0.1, random_state=42)

    # 2. Build uid -> filename for Frontal projections
    image_map = {}
    with open(projections_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            uid = row.get("uid")
            filename = row.get("filename")
            projection = row.get("projection")
            if projection == "Frontal" and uid and filename:
                image_map[str(uid)] = filename

    # 3. Collect allow patterns for snapshot_download
    allow_patterns = []
    for item in val_records:
        uid_str = str(item["uid"])
        if uid_str in image_map:
            filename = image_map[uid_str]
            allow_patterns.append(f"images/images_normalized/{filename}")

    print(f"Prepared {len(allow_patterns)} target validation images for download.")
    if not allow_patterns:
        print("No validation images found. Check projections or reports split.")
        return

    # 4. Run snapshot download
    print("Starting download of validation images from HuggingFace...")
    snapshot_download(
        repo_id="sasi2004/chest-xrays-indiana-university",
        repo_type="dataset",
        allow_patterns=allow_patterns,
        local_dir=str(local_dir)
    )
    print("Download completed successfully!")

if __name__ == "__main__":
    main()
