# Zero-Shot Architecture

This note explains the training-free design choice for the clinical report generator.

## What
A zero-shot, inference-only pipeline combining pre-trained image models (`torchxrayvision`) and cloud LLM APIs (Gemini/Groq) via a retrieval-augmented generation framework. No weights are trained or fine-tuned on the target dataset.

## Why
Fine-tuning domain-specific clinical vision-language models requires substantial GPU compute, high-quality labeled datasets, and risks memorizing/hallucinating clinical text. An inference-only agent pipeline with structured prompting and vector search runs efficiently on a local CPU and yields highly verifiable results.

## Alternatives Considered & Outcome
* **End-to-End Fine-Tuning (e.g. CNN-LSTM, LLaVA)**: Requires heavy GPU training, high risk of hallucinating pathologies, and lacks intermediate clinical reasoning steps.
* **Outcome**: Adopted a modular zero-shot pipeline utilizing public clinical guidelines as context to ground generative outputs.

---
*Related: [[decisions-moc]]*
