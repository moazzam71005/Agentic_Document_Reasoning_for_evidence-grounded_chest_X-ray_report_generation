"""Stage A: schema-constrained structured findings (T=0)."""

from __future__ import annotations

from src.agent.state import EvidenceState
from src.llm.client import LLMClient, parse_json_response

STAGE_A_SCHEMA = {
    "observations": [
        {
            "finding": "string",
            "localization": "string",
            "confidence": "high|moderate|low",
            "evidence_ref": "view_id or feature name",
        }
    ],
    "quality_notes": ["string"],
    "impression_candidates": ["string"],
}


def generate_structured_findings(
    state: EvidenceState,
    llm: LLMClient | None = None,
) -> dict:
    llm = llm or LLMClient()
    prompt = f"""You are a radiology evidence extraction agent.
Given the evidence state below, produce ONLY a JSON object matching this schema:
{STAGE_A_SCHEMA}

Rules:
- Every observation must cite an evidence_ref from the state
- Do NOT invent findings not supported by evidence
- Separate observations from impression_candidates

Evidence state:
{state.to_context()}
"""
    response = llm.chat(
        [{"role": "user", "content": prompt}],
        temperature=0.0,
        json_mode=True,
    )
    return parse_json_response(response)
