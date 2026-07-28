# Ollama Context Window Expansion

This note covers the decision to configure larger context limits for local reasoning LLMs.

## What
Added `"num_ctx": 8192` to the options payload in `_chat_ollama` inside `src/llm/client.py`.

## Why
When prompting reasoning models (like `qwen3.5:2b`), the combined length of the input prompt and generated internal thoughts (within the `<think>` tag) can exceed standard limits. Once exceeded, Ollama terminates generation with `done_reason: length` before the actual response is generated. Expanding the context window to 8192 tokens provides ample headroom for both reasoning chains and the structured JSON output.

## Alternatives Considered & Outcome
* **Aggressive Prompt Pruning**: Truncating prompt elements like patient metadata and evidence indicators. Discouraged as it compromises clinical evidence grounding.
* **Outcome**: Expanded the context size to 8192, enabling stable, untruncated generation on local models.

---
*Related: [[decisions-moc]]*
