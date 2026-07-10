"""Gap analysis Δ_t = GapAnalysis(S_t, C, G(q_t))."""

from __future__ import annotations

from src.agent.checklist import get_checklist
from src.agent.state import EvidenceState


def gap_analysis(
    state: EvidenceState,
    guideline_context: str = "",
) -> list[dict]:
    """Identify checklist items not yet supported by current evidence."""
    gaps = []
    evidence_text = state.to_context().lower()

    for item in get_checklist():
        supported = any(
            key.lower() in evidence_text
            for key in item["evidence_keys"]
        )
        if not supported:
            gaps.append({
                **item,
                "reason": "missing_evidence",
                "guideline_hint": guideline_context[:200] if guideline_context else "",
            })

    return gaps
