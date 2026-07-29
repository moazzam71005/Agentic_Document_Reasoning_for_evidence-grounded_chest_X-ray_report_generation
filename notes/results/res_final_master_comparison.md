# Master Comparison: 3 vs 6 Epochs (Qwen vs Llama)

Presents the final comparative performance benchmarks of all SFT model runs on the 334-record validation split.

Link back to: [[notes/results/results-moc.md|Results MOC]] | [[notes/context/overview.md|Project Overview]]

## 3 vs 6 Epochs Comparison Table

| Metric | Qwen2.5-3B (3 Epochs) | Qwen2.5-3B (6 Epochs) | Llama-3-8B (3 Epochs) | Llama-3-8B (6 Epochs, r=16) | Llama-3-8B (6 Epochs, r=32)* |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **BLEU-1** | **`0.3257`** | `0.3213` | `0.1633` | `0.2769` | `0.1455` |
| **BLEU-4** | `0.1038` | `0.1035` | `0.0509` | **`0.1050`** | `0.0426` |
| **ROUGE-L** | **`0.3456`** | `0.3424` | `0.2060` | `0.3253` | `0.1975` |
| **METEOR** | `0.3025` | `0.2933` | `0.2907` | **`0.3343`** | `0.2673` |
| **BERTScore F1** | **`0.9058`** | `0.8929` | `0.8627` | `0.8837` | `0.8538` |

*\*Note: The Llama-3-8B (r=32) configuration was restricted to attention target modules `["q_proj", "k_proj", "v_proj", "o_proj"]` to fit in the 10GB VRAM ceiling.*

## Best Results Summary Table

| Metric | Best Score | Winning Model Configuration |
| :--- | :---: | :--- |
| **BLEU-1** | `0.3257` | Qwen2.5-3B (3 Epochs, Rank 16, Nucleus Sampling) |
| **BLEU-4** | **`0.1050`** | **Llama-3-OpenBioLLM-8B (6 Epochs, Rank 16, Beam Search + Truncation)** |
| **ROUGE-L** | `0.3456` | Qwen2.5-3B (3 Epochs, Rank 16, Nucleus Sampling) |
| **METEOR** | **`0.3343`** | **Llama-3-OpenBioLLM-8B (6 Epochs, Rank 16, Beam Search + Truncation)** |
| **BERTScore F1**| `0.9058` | Qwen2.5-3B (3 Epochs, Rank 16, Nucleus Sampling) |

## Overall Benchmark Summary
* **Llama-3-8B (6 Epochs, r=16)** achieves the absolute best scores for **BLEU-4** (`0.1050`) and **METEOR** (`0.3343`), representing the highest exact clinical accuracy and term matching.
* **Target Module Coverage is Critical**: Restricting the LoRA adapter to attention projections (in Llama-3 r=32) severely penalizes metric scores (e.g. dropping BLEU-4 to `0.0426`), proving that training on the MLP gate/up/down projections is essential for clinical writing adaptation.
* **Low-Temperature Repetition Bug**: Evaluating quantized 8B models using greedy-like nucleus sampling (`temp=0.1`) without beam search constraints leads to infinite token repetition loops, significantly degrading clinical generation coherence.
* **Qwen2.5-3B (3 Epochs)** retains a small lead in **ROUGE-L** and **BERTScore F1**, which is primarily driven by its low-temperature nucleus sampling generation strategy that avoids the repetition alignment issues sometimes found in beam search decoding constraints.
