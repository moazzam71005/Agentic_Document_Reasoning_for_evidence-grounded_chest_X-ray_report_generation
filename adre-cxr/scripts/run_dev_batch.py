#!/usr/bin/env python3
"""Batch-generate ADRE-CXR reports for the first N studies in a split."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--limit", type=int, default=10, help="Max studies to run")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--sleep", type=float, default=3.0, help="Seconds between studies (rate limit)")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument(
        "--few-shot",
        action="store_true",
        help="Use few-shot Stage B prompting",
    )
    parser.add_argument(
        "--reports-dir",
        default=None,
        help="Directory for .txt reports (default: outputs/reports or outputs/reports_fewshot)",
    )
    parser.add_argument(
        "--predictions",
        default=None,
        help="Output JSONL path (default derived from mode)",
    )
    args = parser.parse_args()

    split = json.loads((ROOT / "data" / "processed" / f"{args.split}_split.json").read_text())
    img_dir = ROOT / "data" / "raw" / "images" / "images_normalized"
    ev_dir = ROOT / "data" / "processed" / "evidence"

    if args.reports_dir:
        reports_dir = ROOT / args.reports_dir
    elif args.few_shot:
        reports_dir = ROOT / "outputs" / "reports_fewshot"
    else:
        reports_dir = ROOT / "outputs" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    if args.predictions:
        pred_path = ROOT / args.predictions
    elif args.few_shot:
        pred_path = ROOT / "outputs" / "fewshot_predictions.jsonl"
    else:
        pred_path = ROOT / "outputs" / f"{args.split}_predictions.jsonl"

    slice_ = split[args.offset : args.offset + args.limit]
    results = []
    if pred_path.exists():
        existing = {}
        for line in pred_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                existing[str(row["uid"])] = row
    else:
        existing = {}

    for i, rec in enumerate(slice_):
        uid = str(rec["uid"])
        out_txt = reports_dir / f"{uid}.txt"
        ev_path = ev_dir / f"{uid}.json"

        if args.skip_existing and out_txt.exists() and out_txt.stat().st_size > 20:
            print(f"[{i+1}/{len(slice_)}] skip existing {uid}")
            existing[uid] = {"uid": uid, "prediction": out_txt.read_text(encoding="utf-8")}
            continue

        if not ev_path.exists():
            print(f"[{i+1}/{len(slice_)}] skip {uid}: no evidence")
            continue

        ev = json.loads(ev_path.read_text(encoding="utf-8"))
        filename = ev.get("filename", "")
        img_path = img_dir / filename
        if not img_path.exists():
            print(f"[{i+1}/{len(slice_)}] skip {uid}: missing image {filename}")
            continue

        indication = rec.get("indication", "") or ev.get("patient_metadata", {}).get("indication", "")
        print(f"[{i+1}/{len(slice_)}] running {uid} ...")
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "run_pipeline.py"),
            "--study-id", uid,
            "--image", str(img_path),
            "--indication", indication,
            "--evidence-json", str(ev_path),
            "--output", str(out_txt),
        ]
        if args.few_shot:
            cmd.append("--few-shot")
        try:
            subprocess.run(cmd, check=True, cwd=str(ROOT))
            text = out_txt.read_text(encoding="utf-8")
            existing[uid] = {"uid": uid, "prediction": text}
            print(f"  OK -> {out_txt}")
        except subprocess.CalledProcessError as e:
            print(f"  FAILED {uid}: exit {e.returncode}")
            time.sleep(max(args.sleep, 20))

        if i < len(slice_) - 1:
            time.sleep(args.sleep)

    with open(pred_path, "w", encoding="utf-8") as f:
        for rec in slice_:
            uid = str(rec["uid"])
            if uid in existing:
                f.write(json.dumps(existing[uid]) + "\n")
                results.append(uid)

    print(f"\nWrote {len(results)} predictions -> {pred_path}")
    print("Evaluate with:")
    print(f"  python scripts/evaluate.py --predictions {pred_path.relative_to(ROOT)} --split {args.split}")


if __name__ == "__main__":
    main()
