#!/usr/bin/env python3
"""Rebuild predictions JSONL from saved report .txt files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--reports-dir", default="outputs/reports")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    split = json.loads((ROOT / "data" / "processed" / f"{args.split}_split.json").read_text(encoding="utf-8"))
    reports_dir = ROOT / args.reports_dir
    out = ROOT / (args.output or f"outputs/{args.split}_predictions.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with open(out, "w", encoding="utf-8") as f:
        for rec in split:
            uid = str(rec["uid"])
            path = reports_dir / f"{uid}.txt"
            if path.exists() and path.stat().st_size > 20:
                f.write(json.dumps({"uid": uid, "prediction": path.read_text(encoding="utf-8")}) + "\n")
                n += 1

    print(f"Wrote {n} predictions -> {out}")


if __name__ == "__main__":
    main()
