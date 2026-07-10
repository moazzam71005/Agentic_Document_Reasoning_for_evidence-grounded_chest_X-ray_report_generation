#!/usr/bin/env python3
"""Build BM25 + FAISS index from guideline PDFs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.knowledge_index.build_index import build_index
from src.knowledge_index.parse_guidelines import parse_guideline_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(ROOT / args.config) as f:
        cfg = yaml.safe_load(f)

    guide_dir = ROOT / cfg["paths"]["guidelines_dir"]
    out_dir = ROOT / cfg["paths"]["processed_dir"] / "index"
    chunks = parse_guideline_dir(guide_dir)
    if not chunks:
        print(f"No PDFs found in {guide_dir}. Download Fleischner glossary first.")
        return
    build_index(chunks, out_dir, cfg["retrieval"]["embedding_model"])
    print(f"Built index with {len(chunks)} chunks → {out_dir}")


if __name__ == "__main__":
    main()
