# Version 3 Llama-3-OpenBioLLM-8B Benchmark Results

Documents the final comparative metrics on the full 334-record validation/test split for the 8B parameter model.

Link back to: [[notes/results/results-moc.md|Results MOC]] | [[notes/versions/v3_llama_openbiollm.md|Version 3: Llama-3 Remote Pipeline]]

## Benchmark Setup
* **Evaluation Dataset**: Full validation/test split of 334 records (Indiana University chest X-rays).
* **Model Configuration**: QLoRA fine-tuned `Llama3-OpenBioLLM-8B` (NF4 4-bit, rank $r=16$, $\alpha=32$, 6 epochs).
* **Inference Strategy**: Beam Search generation (`num_beams=4`, `min_new_tokens=40`, and custom Llama 3 end-of-turn terminators configuration) with `clean_predictions()` post-processing truncation.

## Results Summary Table

| Metric | Zero-Shot (V3) | Few-Shot (V3) | LoRA Fine-Tuned (V3) |
| :--- | :--- | :--- | :--- |
| **BLEU-1** | `0.1455` | `0.1496` | `0.2769` |
| **BLEU-2** | `0.0637` | `0.0607` | `0.1823` |
| **BLEU-3** | `0.0302` | `0.0274` | `0.1344` |
| **BLEU-4** | `0.0138` | `0.0124` | `0.1050` |
| **ROUGE-L** | `0.1526` | `0.1372` | `0.3253` |
| **METEOR** | `0.1866` | `0.1615` | `0.3343` |
| **BERTScore F1** | `0.8467` | `0.8428` | `0.8837` |

## Key Insights
1. **LoRA Fine-Tuning Performance**: QLoRA SFT fine-tuning with 6 epochs and prediction cleanups achieves a **660% relative improvement** in BLEU-4 matching score (`0.1050` vs. `0.0138`) and a **113.1% relative improvement** in ROUGE-L (`0.3253` vs. `0.1526`) over the Zero-Shot baseline.
2. **Semantic Quality**: BERTScore F1 increases significantly from `0.8467` (Zero-Shot) to `0.8837` (LoRA FT), representing highly precise evidence-grounding and consistent radiological terminology.
3. **Boundary Truncation Impact**: Eliminating post-report text generation leaks (such as repeated assistant tags or secondary doctor letters) using the `clean_predictions()` helper dramatically improved the matching scores (increasing BLEU-4 from `0.0509` to `0.1050`).
