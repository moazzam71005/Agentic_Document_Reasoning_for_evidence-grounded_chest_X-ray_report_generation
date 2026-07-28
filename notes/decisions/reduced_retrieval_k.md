# Reduced Retrieval Top-K

This note documents the decision to scale down RAG retrieval settings for local deployment.

## What
Reduced the number of retrieved guideline passages (`top_k`) per iteration from 8 to 2 in `src/agent/controller.py`.

## Why
A `top_k` of 8 pulled in up to 4,000 words (~5,000 tokens) of text. In multi-iteration runs, this context size grows extremely large, causing local models like `qwen3.5:2b` to hit token limits or run out of memory. Lowering the retrieval quantity preserves only the top context chunks, resulting in highly focused prompts that easily fit local LLM context windows.

## Alternatives Considered & Outcome
* **Dynamic Chunk Pruning**: Selectively discarding guidelines using a secondary reranking layer. Rejected due to latency overhead and local compute bounds.
* **Outcome**: Scaled back context retrieval count (`top_k=2`) directly at the retriever interface, reducing overall context size by ~75% and guaranteeing stable execution.

---
*Related: [[decisions-moc]]*
