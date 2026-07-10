"""Report requirement checklist C."""

from __future__ import annotations

REPORT_CHECKLIST = [
    {
        "id": "findings_lungs",
        "section": "findings",
        "requirement": "Describe lung fields (opacity, volume, vasculature)",
        "evidence_keys": ["global_pathology_probs", "regional_burden", "opacity_density"],
    },
    {
        "id": "findings_heart",
        "section": "findings",
        "requirement": "Describe cardiac silhouette and mediastinum",
        "evidence_keys": ["ctr_proxy", "Cardiomegaly"],
    },
    {
        "id": "findings_pleura",
        "section": "findings",
        "requirement": "Comment on pleura and costophrenic angles",
        "evidence_keys": ["Pleural Effusion", "Pneumothorax"],
    },
    {
        "id": "findings_quality",
        "section": "findings",
        "requirement": "Note image quality limitations if present",
        "evidence_keys": ["quality_indicators"],
    },
    {
        "id": "impression_summary",
        "section": "impression",
        "requirement": "Concise diagnostic impression supported by findings",
        "evidence_keys": ["global_abnormality_score", "top_pathology"],
    },
]


def get_checklist() -> list[dict]:
    return REPORT_CHECKLIST.copy()
