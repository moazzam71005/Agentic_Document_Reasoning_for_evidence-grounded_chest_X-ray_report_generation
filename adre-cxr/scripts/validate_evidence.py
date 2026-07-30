#!/usr/bin/env python3
"""Validate one evidence-v2 JSON or every evidence.json below a directory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evidence.schemas import validate_evidence_document


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    args = parser.parse_args()
    target = Path(args.path)
    paths = [target] if target.is_file() else sorted(target.rglob("evidence.json"))
    invalid = 0
    for path in paths:
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
            errors = validate_evidence_document(document)
        except (OSError, ValueError, UnicodeError, TypeError) as exc:
            errors = [f"could not parse: {exc}"]
        if errors:
            invalid += 1
            print(f"INVALID {path}")
            for error in errors:
                print(f"  - {error}")
        else:
            print(f"OK {path}")
    print(json.dumps({"checked": len(paths), "invalid": invalid}, indent=2))
    return 1 if invalid else 0


if __name__ == "__main__":
    raise SystemExit(main())
