# Memory Management Pattern

This note outlines memory optimization patterns for executing pipeline code with local LLMs on resource-constrained environments.

## Context
When running local vision and language models concurrently on a single machine, memory pressure from loaded PyTorch/DenseNet classification weights and SentenceTransformer embedders can exhaust system memory. This causes the Ollama backend runner to fail prompt evaluations or return empty generations.

## Implementation
After the heavy neural classification and embedding retrieval stages complete, reference objects (such as retrievers and controller loops holding the model weights) are explicitly deleted (`del`). The Python garbage collector (`gc.collect()`) is then invoked, and the PyTorch CUDA cache is cleared (`torch.cuda.empty_cache()`) before starting the LLM generation phase.

## Benefits
Ensures that maximum system RAM and VRAM are available for the local LLM runner, preventing resource contention failures and blank token generations.

---
*Related: [[patterns-moc]]*
