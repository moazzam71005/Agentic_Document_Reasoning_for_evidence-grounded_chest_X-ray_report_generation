#!/usr/bin/env python3
"""Score manually verified evidence-review JSONL before report generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _names(items: list[dict]) -> set[str]:
    return {str(item["name"]).strip().lower() for item in items}


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("review_jsonl")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in Path(args.review_jsonl).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = [row for row in rows if row.get("review", {}).get("verified")]
    tp = fp = fn = 0
    negative_correct = negative_total = 0
    uncertainty_correct = uncertainty_total = 0
    location_correct = location_total = 0
    reference_positive_total = extracted_positive_total = 0
    reference_negative_total = extracted_negative_total = 0
    for row in rows:
        extracted_positive = _names(row["extracted"]["positive"])
        extracted_negative = _names(row["extracted"]["negative"])
        extracted_uncertain = _names(row["extracted"]["uncertain"])
        reference_positive = {
            str(x).strip().lower() for x in row["review"]["reference_positive"]
        }
        reference_negative = {
            str(x).strip().lower() for x in row["review"]["reference_negative"]
        }
        reference_uncertain = {
            str(x).strip().lower() for x in row["review"]["reference_uncertain"]
        }
        tp += len(extracted_positive & reference_positive)
        fp += len(extracted_positive - reference_positive)
        fn += len(reference_positive - extracted_positive)
        negative_correct += len(extracted_negative & reference_negative)
        negative_total += len(reference_negative)
        uncertainty_correct += len(extracted_uncertain & reference_uncertain)
        uncertainty_total += len(reference_uncertain)
        reference_positive_total += len(reference_positive)
        extracted_positive_total += len(extracted_positive)
        reference_negative_total += len(reference_negative)
        extracted_negative_total += len(extracted_negative)
        for correct in row["review"].get("evidence_location_correct", {}).values():
            location_total += 1
            location_correct += int(bool(correct))
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    metrics = {
        "verified_studies": len(rows),
        "finding_precision": precision,
        "finding_recall": recall,
        "finding_f1": _ratio(2 * precision * recall, precision + recall),
        "negation_accuracy": _ratio(negative_correct, negative_total),
        "location_accuracy": _ratio(location_correct, location_total),
        "uncertainty_agreement": _ratio(uncertainty_correct, uncertainty_total),
        "positive_finding_coverage": _ratio(tp, reference_positive_total),
        "negative_finding_coverage": _ratio(
            negative_correct, reference_negative_total
        ),
        "counts": {
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "extracted_positive": extracted_positive_total,
            "extracted_negative": extracted_negative_total,
        },
    }
    rendered = json.dumps(metrics, indent=2)
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    print(rendered)
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
