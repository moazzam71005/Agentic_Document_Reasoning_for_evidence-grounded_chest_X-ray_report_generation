#!/usr/bin/env python3
"""Prepare IU X-Ray data: parse reports + create dev/test splits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_prep.make_splits import make_splits, save_splits
from src.data_prep.parse_reports import parse_reports_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, help="Path to IU X-Ray folder")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(ROOT / args.config) as f:
        cfg = yaml.safe_load(f)

    data_root = Path(args.data_root)
    reports = pd.read_csv(data_root / "indiana_reports.csv")
    parsed = parse_reports_df(
        reports,
        min_findings=cfg["data"]["min_findings_chars"],
        min_impression=cfg["data"]["min_impression_chars"],
    )
    print(f"Parsed {len(parsed)} valid reports")

    dev, test = make_splits(
        parsed,
        dev_size=cfg["data"]["dev_size"],
        test_size=cfg["data"]["test_size"],
        seed=cfg["data"]["random_seed"],
    )
    out_dir = ROOT / cfg["paths"]["processed_dir"]
    save_splits(dev, test, out_dir)
    print(f"Saved dev={len(dev)}, test={len(test)} to {out_dir}")


if __name__ == "__main__":
    main()
