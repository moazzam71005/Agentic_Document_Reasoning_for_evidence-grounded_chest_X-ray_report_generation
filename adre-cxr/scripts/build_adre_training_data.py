"""
Build ADRE-style training data: evidence packet + retrieved guidelines → JSONL.

Each training example uses the FULL ADRE context:
  indication + image evidence + guidelines → findings + impression

Run locally (uses the BM25+FAISS index in data/processed/index/):
    python scripts/build_adre_training_data.py \
        --evidence-dir data/processed/evidence \
        --split-file data/processed/dev_split.json \
        --index-dir data/processed/index \
        --output adre_training_data.jsonl \
        --top-k 4
"""

import argparse
import json
import sys
from pathlib import Path

# allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.knowledge_index.retrieve import HybridRetriever


# ──────────────────────────────────────────────────────────────────────────────
# Prompt building helpers
# ──────────────────────────────────────────────────────────────────────────────

def _fmt_pathology(probs: dict, top_n: int = 10) -> str:
    items = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:top_n]
    lines = []
    for label, score in items:
        if not label:
            continue
        bar = "HIGH" if score > 0.65 else ("MOD" if score > 0.55 else "low")
        lines.append(f"  {label:<25} {score:.3f}  [{bar}]")
    return "\n".join(lines) if lines else "  (not available)"


def _fmt_regional(regional: dict) -> str:
    zone_order = [
        "zone_upper_left", "zone_upper_right",
        "zone_mid_left",   "zone_mid_right",
        "zone_lower_left", "zone_lower_right",
    ]
    lines = []
    for zone in zone_order:
        info = regional.get(zone, {})
        if not info:
            continue
        label = zone.replace("zone_", "").replace("_", " ").title()
        top = info.get("top_pathology", "?")
        score = info.get("max_pathology_prob", 0.0)
        lines.append(f"  {label:<20} top={top} ({score:.3f})")
    return "\n".join(lines) if lines else "  (not available)"


def _fmt_features(feats: dict) -> str:
    keys = ["opacity_density", "symmetry_index", "ctr_proxy"]
    lines = []
    for k in keys:
        v = feats.get(k)
        if v is not None:
            lines.append(f"  {k.replace('_', ' ').title():<25} {v:.3f}")
    return "\n".join(lines) if lines else "  (not available)"


def _fmt_quality(qual: dict) -> str:
    keys = ["exposure", "inspiration_adequacy", "rotation_flag"]
    lines = []
    for k in keys:
        v = qual.get(k)
        if v is not None:
            lines.append(f"  {k.replace('_', ' ').title():<30} {v}")
    return "\n".join(lines) if lines else "  (not available)"


def _fmt_guidelines(chunks: list[dict]) -> str:
    parts = []
    for c in chunks:
        title = c.get("title_path", "")
        text = c.get("text", "").strip()[:400]
        parts.append(f"[{title}]\n{text}")
    return "\n\n".join(parts) if parts else "(no guidelines retrieved)"


def build_adre_prompt(
    indication: str,
    evidence: dict,
    guideline_chunks: list[dict],
) -> str:
    """Build the ADRE-style input prompt for one study."""
    return f"""You are a board-certified radiologist. You have access to the patient's clinical indication, AI-extracted image evidence, and relevant radiology guidelines. Use ALL of this information to write a structured chest X-ray report.

=== CLINICAL INDICATION ===
{indication or "Not provided"}

=== AI IMAGE EVIDENCE ===

Pathology Scores (DenseNet121 classifier on frontal view, scale 0–1):
{_fmt_pathology(evidence.get("global_pathology_probs", {}))}

Regional Burden by Lung Zone:
{_fmt_regional(evidence.get("regional_burden", {}))}

Image Features:
{_fmt_features(evidence.get("image_features", {}))}

Quality Indicators:
{_fmt_quality(evidence.get("quality_indicators", {}))}

Global Abnormality Score: {evidence.get("global_abnormality_score", 0.0):.3f}

=== RELEVANT RADIOLOGY GUIDELINES ===
{_fmt_guidelines(guideline_chunks)}

=== TASK ===
Write the report with exactly these two sections:

FINDINGS:
[Detailed paragraph covering lungs (all zones), heart size, mediastinum, pleura, and bones.]

IMPRESSION:
[1–3 sentence clinical summary of key findings and their significance.]"""


def build_target(findings: str, impression: str) -> str:
    """The expected LLM output."""
    return f"FINDINGS:\n{findings.strip()}\n\nIMPRESSION:\n{impression.strip()}"


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence-dir", default="data/processed/evidence",
                    help="Folder containing {uid}.json evidence packets")
    ap.add_argument("--split-file", default="data/processed/dev_split.json",
                    help="JSON array of {uid, indication, findings, impression, ...}")
    ap.add_argument("--index-dir", default="data/processed/index",
                    help="Folder containing bm25.pkl and faiss.index")
    ap.add_argument("--output", default="adre_training_data.jsonl",
                    help="Output JSONL path")
    ap.add_argument("--top-k", type=int, default=4,
                    help="Number of guideline chunks to retrieve per study")
    args = ap.parse_args()

    ev_dir = Path(args.evidence_dir)
    split_path = Path(args.split_file)
    index_dir = Path(args.index_dir)
    out_path = Path(args.output)

    print(f"Loading split from {split_path} ...")
    studies = json.loads(split_path.read_text(encoding="utf-8"))
    print(f"  {len(studies)} studies in split")

    print(f"Loading retriever from {index_dir} ...")
    retriever = HybridRetriever(index_dir)

    written = 0
    skipped = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as fout:
        for study in studies:
            uid = str(study["uid"])
            findings = (study.get("findings") or "").strip()
            impression = (study.get("impression") or "").strip()
            indication = (study.get("indication") or "").strip()

            if not findings or not impression:
                skipped += 1
                continue

            ev_path = ev_dir / f"{uid}.json"
            if not ev_path.exists():
                skipped += 1
                continue

            evidence = json.loads(ev_path.read_text(encoding="utf-8"))

            # build retrieval query from indication + top pathologies
            top_paths = sorted(
                evidence.get("global_pathology_probs", {}).items(),
                key=lambda x: x[1], reverse=True
            )[:5]
            query = indication + " " + " ".join(p for p, _ in top_paths if p)
            chunks = retriever.retrieve(query, top_k=args.top_k)

            prompt = build_adre_prompt(indication, evidence, chunks)
            target = build_target(findings, impression)

            record = {
                "uid": uid,
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": target},
                ]
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"\nDone. Written: {written}  |  Skipped (no evidence / empty): {skipped}")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
