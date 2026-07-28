# Ollama Generation Limit

This note covers the design choice of configuring explicit generation limits for local inference.

## What
Configured an explicit `"num_predict": 4096` token generation limit in the options payload of `src/llm/client.py` for Ollama API requests.

## Why
Reasoning models (such as `qwen3.5:2b`) spend a substantial portion of their token budget outputting internal chains of thought within `<think>` blocks. Under default Ollama configurations, the maximum token generation limit is small, causing the runner to truncate generation before outputting the actual response. Explicitly setting a larger generation window allows both thinking steps and the final JSON response to be successfully outputted.

## Alternatives Considered & Outcome
* **Low Token Budgets**: Restricting the generator to a small window to optimize inference time. This leads to persistent truncation failures on reasoning-heavy prompts.
* **Outcome**: Implemented a default budget of 4096 tokens to successfully support reasoning-based report writing, forced a minimum temperature of 0.8, and added explicit `repeat_penalty: 1.2` and `repeat_last_n: 128` options to prevent local reasoning models from entering repetitive loops during the thinking phase.

---
*Related: [[decisions-moc]]*
