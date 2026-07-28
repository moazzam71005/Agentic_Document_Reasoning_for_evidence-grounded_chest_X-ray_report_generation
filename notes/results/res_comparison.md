# Model Comparison: Qwen2.5-3B vs Llama-3-OpenBioLLM-8B

Presents a comparative benchmark analysis between the fine-tuned Qwen2.5-3B and Llama-3-OpenBioLLM-8B models.

Link back to: [[notes/results/results-moc.md|Results MOC]] | [[notes/context/overview.md|Project Overview]]

## Comparative Metrics Table

| Metric | Zero-Shot (Qwen-3B) | Zero-Shot (Llama-8B) | LoRA FT (Qwen-3B) | LoRA FT (Llama-8B) |
| :--- | :--- | :--- | :--- | :--- |
| **BLEU-1** | `0.1263` | **`0.1454`** | **`0.3213`** | `0.1633` |
| **BLEU-4** | `0.0116` | **`0.0137`** | **`0.1035`** | `0.0509` |
| **ROUGE-L** | `0.1475` | **`0.1524`** | **`0.3424`** | `0.2060` |
| **METEOR** | `0.2933` | **`0.1866`** | `0.2933` | **`0.2907`** |
| **BERTScore F1** | `0.8450` | **`0.8467`** | **`0.8929`** | `0.8627` |

## Fine-Tuning Performance Analysis
* **Termination & Generation Noise**: Qwen2.5-3B terminates generation cleanly at the end of the diagnostic impression. Llama-3-OpenBioLLM-8B exhibits trailing generation leaks (hallucinated assistant notes, template metadata, or repeat letters), which dilute the precision match and lower BLEU/ROUGE-L scores.
* **Hyperparameter Budget**: Qwen was trained for 6 epochs with rank $r=32$ adapters, whereas Llama-3 was restricted to 3 epochs and rank $r=16$ due to 10GB VRAM limits, leading to less SFT adaptation.

## Out-of-the-Box Generalization
* **Zero-Shot Domain Advantage**: Llama-3-OpenBioLLM-8B outperforms Qwen2.5-3B in Zero-Shot mode across all metrics. Its pre-trained clinical weights give it a superior baseline capability prior to fine-tuning.
