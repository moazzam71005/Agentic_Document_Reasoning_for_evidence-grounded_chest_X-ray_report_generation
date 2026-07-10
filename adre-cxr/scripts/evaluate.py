#!/usr/bin/env python3
"""Evaluate generated reports against ground truth."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.metrics import compute_all_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, help="JSONL: {uid, prediction}")
    parser.add_argument("--split", default="test")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(ROOT / args.config) as f:
        cfg = yaml.safe_load(f)

    split = json.loads(
        (ROOT / cfg["paths"]["processed_dir"] / f"{args.split}_split.json").read_text()
    )
    gt = {r["uid"]: r["findings"] + "\n\n" + r["impression"] for r in split}

    preds = []
    refs = []
    with open(args.predictions) as f:
        for line in f:
            row = json.loads(line)
            uid = str(row["uid"])
            if uid in gt:
                preds.append(row["prediction"])
                refs.append(gt[uid])

    metrics = compute_all_metrics(preds, refs, cfg["evaluation"]["bertscore_model"])
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
