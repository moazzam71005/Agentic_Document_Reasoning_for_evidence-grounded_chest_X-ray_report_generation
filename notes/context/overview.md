# ADRE-CXR Project Overview

This vault acts as the persistent context store for the ADRE-CXR project. 

## Project Purpose

**ADRE-CXR** (Agentic Document Reasoning for Chest X-Ray) replicates the ADRE framework (originally built for EEG reports) on chest radiographs. The goal is to generate evidence-grounded clinical report findings and diagnostic impressions from chest X-ray images, using a zero-shot agentic reasoning loop and retrieval-augmented generation (RAG) rather than fine-tuning massive models.

## Core Components

The codebase is organized into four main functional pipelines:
* **Evidence Acquisition Layer**: Generates view crops/enhancements, extracts quality indicators, and predicts pathologies using a frozen image classifier to build an evidence packet.
* **Layout-Aware Knowledge Indexing (RAG)**: Indexes radiology reporting guidelines to provide contextual snippets.
* **Agentic Reasoning Engine**: Leverages an append-only state, clinical requirement checklist, gap analysis, and tool planner to iteratively collect necessary evidence.
* **Dual-Stage Report Generator**: Generates a structured JSON findings schema first, then drafts final narrative prose.

## Tech Stack & Execution Details

* **Core Language**: Python 3.11+
* **Deep Learning Frameworks**: PyTorch, TorchXRayVision (pre-trained DenseNet121)
* **LLM Clients & APIs**: Google Generative AI (Gemini), Groq API, HuggingFace Inference API, Ollama (local)
* **Vector DB & Search**: FAISS and BM25Okapi (sparse-dense hybrid retrieval)
* **Evaluation Stack**: BLEU, ROUGE-L, METEOR, BERTScore

## Vault Indexes (MOCs)

* [[context-moc]]
* [[decisions-moc]]
* [[patterns-moc]]
* [[bugs-moc]]
* [[experiments-moc]]
* [[versions-moc]]
* [[results-moc]]

