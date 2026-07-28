#!/usr/bin/env python3
"""Batch evidence extraction — run on Kaggle GPU."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--skip-neural", action="store_true",
                        help="CPU-only mode: features + quality, no GPU classifier")
    args = parser.parse_args()

    import pandas as pd
    import yaml
    from tqdm import tqdm

    from src.evidence.evidence_packet import build_evidence_packet
    from src.evidence.neural_inference import HAS_TXRV, NeuralInference

    with open(ROOT / args.config) as f:
        cfg = yaml.safe_load(f)

    split_path = ROOT / cfg["paths"]["processed_dir"] / f"{args.split}_split.json"
    records = json.loads(split_path.read_text())
    if args.limit is not None:
        records = records[args.offset : args.offset + args.limit]
    elif args.offset:
        records = records[args.offset:]
    projections = pd.read_csv(Path(args.data_root) / "indiana_projections.csv")

    neural = None
    if not args.skip_neural and HAS_TXRV:
        neural = NeuralInference(weights=cfg["vision"]["classifier_weights"])

    out_dir = ROOT / cfg["paths"]["evidence_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    for rec in tqdm(records, desc="Evidence extraction"):
        uid = rec["uid"]
        proj = projections[projections["uid"] == int(uid)]
        if proj.empty:
            continue
        # Prefer frontal view
        row = proj[proj["projection"] == "Frontal"].iloc[0] if "Frontal" in proj["projection"].values else proj.iloc[0]
        img_name = row["filename"]
        img_path = Path(args.data_root) / "images" / "images_normalized" / img_name
        if not img_path.exists():
            img_path = Path(args.data_root) / "images" / img_name
        if not img_path.exists():
            continue

        packet = build_evidence_packet(
            study_id=str(uid),
            image_path=str(img_path),
            patient_metadata={"indication": rec.get("indication", "")},
            neural=neural,
            rows=cfg["vision"]["zone_rows"],
            cols=cfg["vision"]["zone_cols"],
        )
        packet.save(out_dir / f"{uid}.json")

    print(f"Saved evidence packets to {out_dir}")


if __name__ == "__main__":
    main()
