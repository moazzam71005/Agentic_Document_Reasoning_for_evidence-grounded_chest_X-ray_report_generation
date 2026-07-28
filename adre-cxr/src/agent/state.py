"""Append-only evidence state S_t."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvidenceState:
    patient_metadata: dict[str, Any] = field(default_factory=dict)
    evidence_items: list[dict[str, Any]] = field(default_factory=list)
    guideline_passages: list[dict[str, Any]] = field(default_factory=list)
    iteration: int = 0

    def append_evidence(self, item: dict[str, Any]) -> None:
        self.evidence_items.append(item)

    def append_guidelines(self, passages: list[dict[str, Any]]) -> None:
        seen_ids = {g["id"] for g in self.guideline_passages if "id" in g}
        for p in passages:
            if p.get("id") not in seen_ids:
                self.guideline_passages.append(p)

    def to_context(self) -> str:
        lines = ["=== Patient ===", str(self.patient_metadata)]
        lines.append("\n=== Evidence (append-only) ===")
        for i, e in enumerate(self.evidence_items):
            lines.append(f"[{i}] {e}")
        lines.append("\n=== Retrieved Guidelines ===")
        for g in self.guideline_passages:
            lines.append(g.get("text", str(g)))
        return "\n".join(lines)
