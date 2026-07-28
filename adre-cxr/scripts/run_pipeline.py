#!/usr/bin/env python3
"""Single-record end-to-end ADRE-CXR pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent.controller import AgentController
from src.evidence.evidence_packet import EvidencePacket
from src.generation.repair import repair_report, validate_report
from src.generation.stage_a_findings import generate_structured_findings
from src.generation.stage_b_narrative import generate_narrative_report
from src.knowledge_index.retrieve import HybridRetriever
from src.llm.client import LLMClient


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ADRE-CXR on one study")
    parser.add_argument("--study-id", required=True)
    parser.add_argument("--image", required=True, help="Path to chest X-ray PNG")
    parser.add_argument("--indication", default="")
    parser.add_argument("--evidence-json", default=None, help="Precomputed evidence packet")
    parser.add_argument("--index-dir", default="data/processed/index")
    parser.add_argument("--no-rag", action="store_true")
    parser.add_argument("--no-agentic", action="store_true")
    parser.add_argument("--skip-neural", action="store_true")
    parser.add_argument("--output", default="output_report.txt")
    args = parser.parse_args()

    llm = LLMClient()
    retriever = None
    index_path = ROOT / args.index_dir
    if not args.no_rag and (index_path / "bm25.pkl").exists():
        retriever = HybridRetriever(index_path)

    precomputed = None
    if args.evidence_json:
        precomputed = EvidencePacket.load(Path(args.evidence_json))

    if args.no_agentic:
        from src.evidence.evidence_packet import build_evidence_packet
        from src.agent.state import EvidenceState
        packet = precomputed or build_evidence_packet(
            args.study_id, args.image, {"indication": args.indication}
        )
        state = EvidenceState(patient_metadata={"indication": args.indication})
        state.append_evidence(packet.to_dict())
    else:
        controller = AgentController(
            retriever=retriever,
            use_neural=precomputed is None and not args.skip_neural,
        )
        state = controller.run(
            args.study_id, args.image,
            {"indication": args.indication},
            precomputed_packet=precomputed,
        )

    structured = generate_structured_findings(state, llm)
    report = generate_narrative_report(structured, state, llm)
    if validate_report(report):
        report = repair_report(report, structured, state, llm)

    out = Path(args.output)
    out.write_text(report, encoding="utf-8")
    print(json.dumps({"structured": structured, "report_path": str(out)}, indent=2))


if __name__ == "__main__":
    main()
