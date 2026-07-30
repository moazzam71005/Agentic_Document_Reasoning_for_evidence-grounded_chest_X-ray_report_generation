#!/usr/bin/env python3
"""Create a 50-100-study manual evidence review JSONL template."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--size", type=int, default=75)
    parser.add_argument("--output", default="evidence_review_set.jsonl")
    args = parser.parse_args()
    if not 50 <= args.size <= 100:
        raise SystemExit("--size must be between 50 and 100")
    records = json.loads(Path(args.split_file).read_text(encoding="utf-8"))[: args.size]
    evidence_root = Path(args.evidence_dir)
    output = Path(args.output)
    created = 0
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            uid = str(record["uid"])
            evidence_path = evidence_root / uid / "evidence.json"
            if not evidence_path.exists():
                continue
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            groups = evidence["fused_findings"]
            row = {
                "uid": uid,
                "reference": {
                    "findings": record.get("findings", ""),
                    "impression": record.get("impression", ""),
                },
                "extracted": {
                    group: [
                        {
                            "name": item["name"],
                            "status": item["status"],
                            "score": item["score"],
                            "reliability": item["reliability_level"],
                        }
                        for item in groups[group]
                    ]
                    for group in (
                        "positive",
                        "negative",
                        "uncertain",
                        "not_assessable",
                    )
                },
                "review": {
                    "verified": False,
                    "reference_positive": [],
                    "reference_negative": [],
                    "reference_uncertain": [],
                    "reference_locations": {},
                    "evidence_location_correct": {},
                    "notes": "",
                },
            }
            handle.write(json.dumps(row) + "\n")
            created += 1
    print(json.dumps({"created": created, "output": str(output)}, indent=2))
    return 0 if created else 1


if __name__ == "__main__":
    raise SystemExit(main())
