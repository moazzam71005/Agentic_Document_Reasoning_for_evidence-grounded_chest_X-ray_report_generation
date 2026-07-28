# Model Comparison: Qwen2.5-3B vs Llama-3-OpenBioLLM-8B

Presents a comparative benchmark analysis between the fine-tuned Qwen2.5-3B and Llama-3-OpenBioLLM-8B models.

Link back to: [[notes/results/results-moc.md|Results MOC]] | [[notes/context/overview.md|Project Overview]]

## Comparative Metrics Table

| Metric | Zero-Shot (Qwen-3B) | Zero-Shot (Llama-8B) | LoRA FT (Qwen-3B) | LoRA FT (Llama-8B) |
| :--- | :--- | :--- | :--- | :--- |
| **BLEU-1** | `0.1263` | **`0.1455`** | **`0.3213`** | `0.2769` |
| **BLEU-4** | `0.0116` | **`0.0138`** | `0.1035` | **`0.1050`** |
| **ROUGE-L** | `0.1475` | **`0.1526`** | **`0.3424`** | `0.3253` |
| **METEOR** | `0.1834` | **`0.1866`** | `0.2933` | **`0.3343`** |
| **BERTScore F1** | `0.8450` | **`0.8467`** | **`0.8929`** | `0.8837` |

## Fine-Tuning Performance Analysis
* **Optimized Alignment**: By resolving tokenizer and generation leaks (stripping out duplicate headers and extra conversational fragments after the diagnostic impression), the fine-tuned **Llama-3-OpenBioLLM-8B** now outperforms **Qwen2.5-3B** in exact n-gram clinical match metrics, achieving a BLEU-4 of **`0.1050`** (vs Qwen's `0.1035`) and a METEOR score of **`0.3343`** (vs Qwen's `0.2933`).
* **Semantic Grounding**: Qwen2.5-3B still holds a tiny edge in ROUGE-L (`0.3424` vs `0.3253`) and BERTScore F1 (`0.8929` vs `0.8837`), likely due to having double the adapter rank capacity ($r=32$ vs $r=16$) and therefore fitting the dataset's sentence structural patterns slightly closer.

## Out-of-the-Box Generalization
* **Zero-Shot Domain Advantage**: Llama-3-OpenBioLLM-8B outperforms Qwen2.5-3B in Zero-Shot mode across all metrics. Its pre-trained clinical weights give it a superior baseline capability prior to fine-tuning.
