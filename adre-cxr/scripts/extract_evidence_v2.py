#!/usr/bin/env python3
"""Extract modular, multi-view evidence-v2 packets for an IU-Xray split."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, help="IU-Xray dataset directory")
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--split-file", default=None, help="Optional explicit split JSON")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--config", default="configs/evidence_v2.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"])
    parser.add_argument("--skip-models", action="store_true")
    parser.add_argument("--force", action="store_true", help="Ignore matching cache entries")
    parser.add_argument(
        "--ablation",
        type=int,
        choices=range(1, 8),
        default=None,
        help="Apply controlled evidence ablation 1-7",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        from tqdm import tqdm

        from src.evidence.classifiers import build_models
        from src.evidence.config import apply_ablation, load_evidence_config
        from src.evidence.extractor import EvidenceExtractorV2
        from src.evidence.study_loader import IUStudyLoader

        config_path = _project_path(args.config)
        config = load_evidence_config(config_path)
        if args.ablation is not None:
            config = apply_ablation(config, args.ablation)
        data_root = Path(args.data_root)
        split_path = (
            Path(args.split_file)
            if args.split_file
            else ROOT / "data" / "processed" / f"{args.split}_split.json"
        )
        if not split_path.exists():
            raise FileNotFoundError(
                f"Split file not found: {split_path}. Run scripts/prepare_data.py first."
            )
        records = json.loads(split_path.read_text(encoding="utf-8"))
        records = records[args.offset :]
        if args.limit is not None:
            records = records[: args.limit]
        output_dir = _project_path(
            args.output_dir or config["paths"]["output_dir"]
        )
        models = (
            []
            if args.skip_models or not config["classification"].get("enabled", True)
            else build_models(config["classification"], device=args.device)
        )
        loader = IUStudyLoader(data_root)
        extractor = EvidenceExtractorV2(config, models=models)
    except Exception as exc:
        logging.getLogger(__name__).exception("Fatal configuration/model error")
        print(json.dumps({"status": "fatal", "error": str(exc)}, indent=2))
        return 2

    counts = {"success": 0, "partial_failure": 0, "failed": 0, "cache_hits": 0}
    batch_errors: list[dict[str, str]] = []
    for record in tqdm(records, desc="Evidence v2 extraction"):
        uid = str(record["uid"])
        try:
            document = extractor.extract(
                loader.load(uid, metadata=record),
                output_dir,
                force=args.force,
            )
            counts[document["processing_status"]] += 1
            counts["cache_hits"] += int(bool(document.pop("_cache_hit", False)))
        except Exception as exc:
            logging.getLogger(__name__).exception("Study %s failed", uid)
            counts["failed"] += 1
            batch_errors.append(
                {"uid": uid, "type": type(exc).__name__, "message": str(exc)}
            )

    summary = {
        "requested": len(records),
        **counts,
        "output_dir": str(output_dir),
        "batch_errors": batch_errors,
    }
    (output_dir / "batch_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 1 if counts["partial_failure"] or counts["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
