# Dual-Stage Generation

This note details the split between structured findings and narrative generation.

## What
The report generation process is split into two distinct steps:
1. **Stage A**: Generates a schema-constrained JSON representing structured clinical findings at temperature `T=0.0`.
2. **Stage B**: Translates the findings JSON + retrieved guidelines into a natural language narrative (Findings & Impression) at temperature `T=0.25`.

## Why
Generating free-form narratives directly from evidence packets is highly prone to omission of negative findings, inconsistent localization, and formatting errors. Forcing the LLM to write a structured intermediate JSON ensures all observations are systematically logged and validated prior to writing narrative paragraphs.

## Alternatives Considered & Outcome
* **Direct Narrative Generation**: Generating the paragraphs directly from raw evidence text. This led to formatting issues and higher hallucination rates in baseline testing.
* **Outcome**: Implemented the dual-stage generator to enforce high clinical accuracy and layout consistency, backed by a validation-repair loop.

---
*Related: [[decisions-moc]]*
