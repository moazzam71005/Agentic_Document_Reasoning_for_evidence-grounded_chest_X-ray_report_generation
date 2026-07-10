"""Ablation variant matrix (Table 1 style)."""

from __future__ import annotations

from enum import Enum


class Variant(str, Enum):
    BASELINE = "baseline"
    AGENTIC_ONLY = "agentic_only"
    RAG_ONLY = "rag_only"
    AGENTIC_DUAL = "agentic_dual"
    RAG_AGENTIC_SINGLE = "rag_agentic_single"
    FULL_ADRE = "full_adre"


VARIANT_CONFIG = {
    Variant.BASELINE: {"agentic": False, "rag": False, "dual_stage": False},
    Variant.AGENTIC_ONLY: {"agentic": True, "rag": False, "dual_stage": False},
    Variant.RAG_ONLY: {"agentic": False, "rag": True, "dual_stage": False},
    Variant.AGENTIC_DUAL: {"agentic": True, "rag": False, "dual_stage": True},
    Variant.RAG_AGENTIC_SINGLE: {"agentic": True, "rag": True, "dual_stage": False},
    Variant.FULL_ADRE: {"agentic": True, "rag": True, "dual_stage": True},
}
