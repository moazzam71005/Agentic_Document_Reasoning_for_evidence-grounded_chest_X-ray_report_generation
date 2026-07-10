"""PDF guideline parsing into hierarchical chunks."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


def parse_pdf_simple(pdf_path: Path) -> list[dict[str, Any]]:
    """
    Lightweight PDF parser using PyMuPDF.
    For layout-aware parsing matching ADRE exactly, swap in MinerU later.
    """
    import fitz  # pymupdf

    doc = fitz.open(pdf_path)
    chunks = []
    current_title = pdf_path.stem

    for page_num, page in enumerate(doc):
        text = page.get_text()
        paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
        for idx, para in enumerate(paragraphs):
            if len(para) < 40:
                if para.isupper() or re.match(r"^\d+\.", para):
                    current_title = para
                continue
            chunks.append({
                "id": f"{pdf_path.stem}_p{page_num}_c{idx}",
                "parent_id": pdf_path.stem,
                "title_path": f"{pdf_path.stem} > {current_title}",
                "page_span": [page_num, page_num],
                "sibling_idx": idx,
                "text": para,
            })
    return chunks


def parse_guideline_dir(guidelines_dir: Path) -> list[dict[str, Any]]:
    all_chunks = []
    for pdf in sorted(guidelines_dir.glob("*.pdf")):
        all_chunks.extend(parse_pdf_simple(pdf))
    return all_chunks
