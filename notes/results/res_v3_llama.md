# Version 3 Llama-3-OpenBioLLM-8B Benchmark Results

Documents the final comparative metrics on the full 334-record validation/test split for the 8B parameter model.

Link back to: [[notes/results/results-moc.md|Results MOC]] | [[notes/versions/v3_llama_openbiollm.md|Version 3: Llama-3 Remote Pipeline]]

## Benchmark Setup
* **Evaluation Dataset**: Full validation/test split of 334 records (Indiana University chest X-rays).
* **Model Configuration**: QLoRA fine-tuned `Llama3-OpenBioLLM-8B` (NF4 4-bit, rank $r=16$, $\alpha=32$, 3 epochs).
* **Inference Strategy**: Beam Search generation (`num_beams=4`, `min_new_tokens=40`, and custom Llama 3 end-of-turn terminators configuration).

## Results Summary Table

| Metric | Zero-Shot (V3) | Few-Shot (V3) | LoRA Fine-Tuned (V3) |
| :--- | :--- | :--- | :--- |
| **BLEU-1** | `0.1454` | `0.1496` | `0.1633` |
| **BLEU-2** | `0.0637` | `0.0606` | `0.1027` |
| **BLEU-3** | `0.0302` | `0.0274` | `0.0705` |
| **BLEU-4** | `0.0137` | `0.0124` | `0.0509` |
| **ROUGE-L** | `0.1524` | `0.1372` | `0.2060` |
| **METEOR** | `0.1866` | `0.1615` | `0.2907` |
| **BERTScore F1** | `0.8467` | `0.8428` | `0.8627` |

## Key Insights
1. **LoRA Fine-Tuning Performance**: QLoRA SFT fine-tuning achieves a **271.5% relative improvement** in BLEU-4 matching score (`0.0509` vs. `0.0137`) and a **35.1% relative improvement** in ROUGE-L (`0.2060` vs. `0.1524`) over the Zero-Shot baseline.
2. **Semantic Quality**: BERTScore F1 increases significantly from `0.8467` (Zero-Shot) to `0.8627` (LoRA FT), representing highly precise evidence-grounding and consistent radiological terminology.
3. **Few-Shot Comparison**: Few-Shot prompting degrades slightly compared to Zero-Shot (`0.0124` vs. `0.0137` BLEU-4), demonstrating that in-context prompting on complex clinical formats can cause search convergence issues in constrained decoder pipelines.
