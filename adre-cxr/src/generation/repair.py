"""Format validation and repair prompts."""

from __future__ import annotations

import re

from src.agent.state import EvidenceState
from src.llm.client import LLMClient


def validate_report(text: str) -> list[str]:
    errors = []
    
    # Reject prompt leaks and conversational helper text
    leak_phrases = [
        "formatting errors",
        "repair it to include",
        "do not add unsupported",
        "is classified as",
        "detailed findings paragraph",
        "concise impression paragraph",
        "should not output",
        "should be output",
        "thinking process",
        "internal monologue",
        "is the corrected",
        "the answer is",
        "original report",
        "ground truth",
        "evidence state",
        "board-certified radiologist"
    ]
    
    lower_text = text.lower()
    for phrase in leak_phrases:
        if phrase in lower_text:
            errors.append(f"prompt_leak_detected_{phrase.replace(' ', '_')}")
            break
            
    if not re.search(r"(?i)findings\s*:", text):
        errors.append("missing_findings_section")
    if not re.search(r"(?i)impression\s*:", text):
        errors.append("missing_impression_section")
    return errors


def repair_report(
    report: str,
    structured_findings: dict,
    state: EvidenceState,
    llm: LLMClient | None = None,
) -> str:
    llm = llm or LLMClient()
    errors = validate_report(report)
    if not errors:
        return report

    prompt = f"""IMPORTANT: Do NOT output any thinking process, internal monologue, or '<think>' tags. Do NOT write any reasoning steps. Skip thinking entirely and output ONLY the corrected report.

The following chest X-ray report has formatting errors: {errors}

Repair it to include FINDINGS: and IMPRESSION: sections.
Do not add unsupported content.

Original report:
{report}

Structured findings (ground truth constraints):
{structured_findings}

Evidence:
{state.to_context()}
"""
    return llm.chat([{"role": "user", "content": prompt}], temperature=0.0)
