"""Stage B: narrative Findings + Impression (T=0.25)."""

from __future__ import annotations

from src.agent.state import EvidenceState
from src.llm.client import LLMClient


def generate_narrative_report(
    structured_findings: dict,
    state: EvidenceState,
    llm: LLMClient | None = None,
) -> str:
    llm = llm or LLMClient()
    prompt = f"""You are a board-certified radiologist writing a chest X-ray report.

Write two sections exactly:
FINDINGS:
<detailed findings paragraph>

IMPRESSION:
<concise impression paragraph>

Rules:
- Use only findings supported by the structured evidence below
- Follow standard radiology phrasing from retrieved guidelines
- Do not add unsupported diagnoses

Structured findings:
{structured_findings}

Evidence and guidelines:
{state.to_context()}
"""
    return llm.chat(
        [{"role": "user", "content": prompt}],
        temperature=0.25,
        top_p=0.9,
    )
