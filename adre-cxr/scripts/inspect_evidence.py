#!/usr/bin/env python3
"""Print a concise human-audit view of one evidence-v2 study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence_json")
    args = parser.parse_args()
    path = Path(args.evidence_json)
    document = json.loads(path.read_text(encoding="utf-8"))
    print(f"UID: {document['uid']}")
    print(f"Status: {document['processing_status']}")
    print(f"Quality: {document['study_quality']}")
    print("Views:")
    for view in document["views"]:
        print(
            f"  {view['view_id']}: {view['view_type']} "
            f"role={view['role']} quality={view.get('validation', {}).get('quality_score', 0):.2f}"
        )
    for group in ("positive", "negative", "uncertain", "not_assessable"):
        print(f"{group}:")
        for item in document["fused_findings"][group]:
            print(
                f"  {item['name']}: {item['status']} score={item['score']:.3f} "
                f"reliability={item['reliability_level']}"
            )
    print(f"Visual summary: {path.parent / 'prediction_summary.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
