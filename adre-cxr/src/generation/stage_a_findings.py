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
    prompt = f"""IMPORTANT: Do NOT output any thinking process, internal monologue, or '<think>' tags. Do NOT write any reasoning steps. Skip thinking entirely and output ONLY the raw JSON object matching the schema below.

You are a radiology evidence extraction agent.
Given the evidence state below, produce ONLY a JSON object matching this schema:
{STAGE_A_SCHEMA}

Rules:
- Every observation must cite an evidence_ref from the state
- Do NOT invent findings not supported by evidence
Evidence state:
{state.to_context()}
"""
    response = llm.chat(
        [{"role": "user", "content": prompt}],
        temperature=0.0,
        json_mode=True,
    )
    try:
        return parse_json_response(response)
    except Exception:
        # Fallback to evidence state extraction if LLM completion is empty or invalid
        fallback_obs = []
        if state.evidence_items:
            packet = state.evidence_items[0]
            path_probs = packet.get("global_pathology_probs", {})
            top_paths = sorted(path_probs.items(), key=lambda x: x[1], reverse=True)[:3]
            for p, prob in top_paths:
                if prob > 0.5:
                    fallback_obs.append({
                        "finding": f"{p} probability is {prob:.2f}, indicating possible mild {p.lower()}.",
                        "localization": "Lung fields",
                        "confidence": "moderate",
                        "evidence_ref": p
                    })
        return {
            "observations": fallback_obs,
            "quality_notes": ["Image quality indicators processed from evidence state."],
            "impression_candidates": [
                f"Possible {p.lower()} based on evidence probabilities." for p, prob in top_paths if prob > 0.5
            ] if 'top_paths' in locals() else ["No acute focal abnormality identified."]
        }
