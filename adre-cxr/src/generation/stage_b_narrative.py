"""Stage B: narrative Findings + Impression (T=0.25)."""

from __future__ import annotations

from src.agent.state import EvidenceState
from src.llm.client import LLMClient

# Fixed examples from IU X-Ray training set (UIDs 5, 45, 7) — outside frozen
# dev/test splits, so there is no evaluation leakage.
FEW_SHOT_EXAMPLES = """
Here are example chest X-ray reports in the expected style:

Example 1 (Normal):
Indication: Chest and nasal congestion.
FINDINGS:
The cardiomediastinal silhouette and pulmonary vasculature are within normal limits. There is no pneumothorax or pleural effusion. There are no focal areas of consolidation. Cholecystectomy clips are present. Small T-spine osteophytes. There is biapical pleural thickening, unchanged from prior. Mildly hyperexpanded lungs.
IMPRESSION:
No acute cardiopulmonary abnormality.

Example 2 (Cardiomegaly):
Indication: Edema.
FINDINGS:
Stable cardiomegaly with vascular prominence without overt edema. No focal airspace disease. No large pleural effusion or pneumothorax. The osseous structures are intact.
IMPRESSION:
Stable cardiomegaly without overt pulmonary edema.

Example 3 (Atelectasis):
Indication: Preop lumbar surgery
FINDINGS:
The cardiac contours are normal. Basilar atelectasis. The lungs are clear. Thoracic spondylosis. Lower cervical arthritis.
IMPRESSION:
Basilar atelectasis. No confluent lobar consolidation or pleural effusion.
"""


def generate_narrative_report(
    structured_findings: dict,
    state: EvidenceState,
    llm: LLMClient | None = None,
    use_few_shot: bool = False,
) -> str:
    llm = llm or LLMClient()
    few_shot_block = FEW_SHOT_EXAMPLES if use_few_shot else ""
    prompt = f"""You are a board-certified radiologist writing a chest X-ray report.
{few_shot_block}
Write two sections exactly:
FINDINGS:
<detailed findings paragraph>

IMPRESSION:
<concise impression paragraph>

Rules:
- Use only findings supported by the structured evidence below
- Follow standard radiology phrasing from retrieved guidelines
- Do not add unsupported diagnoses
- Match the brevity and style of the examples when provided

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
