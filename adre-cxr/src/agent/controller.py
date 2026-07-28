"""Main agentic loop — Algorithm 1 equivalent."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.agent.gap_analysis import gap_analysis
from src.agent.planner import plan_tools
from src.agent.state import EvidenceState
from src.agent.consistency import sanity_check
from src.evidence.evidence_packet import EvidencePacket, build_evidence_packet
from src.evidence.neural_inference import NeuralInference
from src.evidence.view_builder import build_views
from src.evidence.image_features import extract_features
from src.knowledge_index.retrieve import HybridRetriever


class AgentController:
    def __init__(
        self,
        retriever: HybridRetriever | None = None,
        tmax: int = 12,
        use_neural: bool = False,
    ):
        self.retriever = retriever
        self.tmax = tmax
        self.use_neural = use_neural
        self._neural = NeuralInference() if use_neural else None

    def run(
        self,
        study_id: str,
        image_path: str,
        patient_metadata: dict | None = None,
        precomputed_packet: EvidencePacket | None = None,
    ) -> EvidenceState:
        state = EvidenceState(patient_metadata=patient_metadata or {})
        views_used: set[str] = set()

        if precomputed_packet:
            state.append_evidence(precomputed_packet.to_dict())
            views_used.add("original")
        else:
            packet = build_evidence_packet(
                study_id, image_path, patient_metadata, neural=self._neural
            )
            state.append_evidence(packet.to_dict())
            views_used.add("original")

        for t in range(self.tmax):
            state.iteration = t
            query = self._form_query(state)
            guideline_context = ""
            if self.retriever:
                hits = self.retriever.retrieve(query, top_k=2)
                state.append_guidelines(hits)
                guideline_context = " ".join(h["text"] for h in hits)

            gaps = gap_analysis(state, guideline_context)
            if not gaps:
                break

            requests = plan_tools(gaps, views_used)
            if not requests:
                break

            for req in requests:
                vid = req["view_id"]
                if vid in views_used:
                    continue
                views_used.add(vid)
                views = build_views(image_path)
                match = next((v for v in views if v.view_id == vid), None)
                if match:
                    item = {
                        "view_id": vid,
                        "image_features": extract_features(match.image),
                        "provenance": {"view_id": vid, "tool": req["tool"]},
                    }
                    if self._neural:
                        probs = self._neural.predict(match.image)
                        item["pathology_probs"] = probs
                    state.append_evidence(item)

            warnings = sanity_check(state.evidence_items)
            if warnings and t < self.tmax - 1:
                continue
            if not gaps:
                break

        return state

    def _form_query(self, state: EvidenceState) -> str:
        meta = state.patient_metadata
        indication = meta.get("indication", "")
        return f"chest x-ray findings impression {indication} lung heart pleura"
