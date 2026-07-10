"""IU X-Ray report parsing."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd


SECTION_PATTERNS = {
    "comparison": r"(?i)(comparison|comp)[:\s]+(.*?)(?=indication|findings|impression|$)",
    "indication": r"(?i)(indication|ind)[:\s]+(.*?)(?=comparison|findings|impression|$)",
    "findings": r"(?i)(findings|finding)[:\s]+(.*?)(?=impression|comparison|indication|$)",
    "impression": r"(?i)(impression|imp)[:\s]+(.*?)(?=findings|comparison|indication|$)",
}


def _extract_section(text: str, pattern: str) -> str:
    if not isinstance(text, str):
        return ""
    match = re.search(pattern, text, re.DOTALL)
    return match.group(2).strip() if match else ""


def parse_report_row(row: pd.Series) -> dict[str, Any]:
    """Parse one IU X-Ray report row into structured sections."""
    # Kaggle CSV often has dedicated columns
    if "findings" in row.index and "impression" in row.index:
        findings = str(row.get("findings", "")).strip()
        impression = str(row.get("impression", "")).strip()
        comparison = str(row.get("comparison", "")).strip()
        indication = str(row.get("indication", "")).strip()
    else:
        full = str(row.get("report", row.get("Report", "")))
        findings = _extract_section(full, SECTION_PATTERNS["findings"])
        impression = _extract_section(full, SECTION_PATTERNS["impression"])
        comparison = _extract_section(full, SECTION_PATTERNS["comparison"])
        indication = _extract_section(full, SECTION_PATTERNS["indication"])

    uid = str(row.get("uid", row.get("id", row.name)))
    return {
        "uid": uid,
        "comparison": comparison,
        "indication": indication,
        "findings": findings,
        "impression": impression,
    }


def is_valid_report(parsed: dict, min_findings: int = 10, min_impression: int = 5) -> bool:
    return (
        len(parsed.get("findings", "")) >= min_findings
        and len(parsed.get("impression", "")) >= min_impression
    )


def parse_reports_df(df: pd.DataFrame, **kwargs) -> list[dict]:
    records = [parse_report_row(row) for _, row in df.iterrows()]
    return [r for r in records if is_valid_report(r, **kwargs)]
