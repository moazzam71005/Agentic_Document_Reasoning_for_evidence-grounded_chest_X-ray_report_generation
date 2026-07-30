#!/usr/bin/env python3
"""Compare legacy evidence packets with their evidence-v2 counterparts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _legacy_paths(root: Path) -> dict[str, Path]:
    return {path.stem: path for path in root.glob("*.json")}


def _v2_paths(root: Path) -> dict[str, Path]:
    return {path.parent.name: path for path in root.glob("*/evidence.json")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1-dir", required=True)
    parser.add_argument("--v2-dir", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    v1, v2 = _legacy_paths(Path(args.v1_dir)), _v2_paths(Path(args.v2_dir))
    rows: list[dict] = []
    for uid in sorted(v1.keys() & v2.keys()):
        old = json.loads(v1[uid].read_text(encoding="utf-8"))
        new = json.loads(v2[uid].read_text(encoding="utf-8"))
        old_scores = old.get("global_pathology_probs", {})
        new_findings = [
            item
            for group in new.get("fused_findings", {}).values()
            for item in group
        ]
        new_scores = {item["name"]: item.get("score", 0.0) for item in new_findings}
        labels = old_scores.keys() & new_scores.keys()
        mean_delta = (
            sum(abs(float(old_scores[x]) - float(new_scores[x])) for x in labels)
            / len(labels)
            if labels
            else None
        )
        rows.append(
            {
                "uid": uid,
                "v1_labels": len(old_scores),
                "v2_labels": len(new_scores),
                "mean_absolute_score_delta": mean_delta,
                "v2_positive": len(new["fused_findings"]["positive"]),
                "v2_negative": len(new["fused_findings"]["negative"]),
                "v2_uncertain": len(new["fused_findings"]["uncertain"]),
                "v2_not_assessable": len(
                    new["fused_findings"]["not_assessable"]
                ),
                "v2_quality": new.get("study_quality", {}).get("overall_score"),
                "v2_conflicts": sum(
                    check.get("type") == "view_conflict"
                    for check in new.get("consistency_checks", [])
                ),
            }
        )
    result = {
        "paired_studies": len(rows),
        "only_v1": sorted(v1.keys() - v2.keys()),
        "only_v2": sorted(v2.keys() - v1.keys()),
        "studies": rows,
    }
    rendered = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
