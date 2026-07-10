"""Format validation and repair prompts."""

from __future__ import annotations

import re

from src.agent.state import EvidenceState
from src.llm.client import LLMClient


def validate_report(text: str) -> list[str]:
    errors = []
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

    prompt = f"""The following chest X-ray report has formatting errors: {errors}

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
