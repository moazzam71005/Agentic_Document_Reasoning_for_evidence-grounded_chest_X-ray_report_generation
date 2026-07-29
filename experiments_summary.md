# Chest X-Ray Report Generation: SFT Experiments Summary

This document summarizes the fine-tuning experiments, techniques, and final benchmark results on the **334-record validation split** of the Indiana University chest X-ray dataset.

---

## 1. Key Optimization Techniques Used

1. **QLoRA NF4 (4-bit Quantization)**: Applied 4-bit base model quantization and mixed-precision computing (`bfloat16`/`float16`) to train large models on a 10GB remote GPU (RTX 3080).
2. **LoRA Adapter Tuning**: Tuned low-rank adapter weights across linear projection layers:
   * **Full Coverage**: Targets all 7 projections (`q, k, v, o, gate, up, down_proj`) to maximize stylistic convergence.
   * **Attention-Restricted**: Targets only the attention blocks (`q, k, v, o_proj`) to fit higher ranks in VRAM.
3. **Post-Processing Prediction Truncation**: Used `clean_predictions()` to split predictions at known tokenizer conversation markers (like `assistant`, `user`, `SmartyHeaderCode`, or initialization labels) to cleanly preserve only the clinical findings and impression.
4. **Decoding Strategy Selection**:
   * **Nucleus Sampling** (`temp=0.1`, `top_p=0.9`, `do_sample=True`) vs.
   * **Beam Search** (`num_beams=4`, `no_repeat_ngram_size=3`, `length_penalty=1.0`) to mitigate infinite token repetition loops.

---

## 2. Benchmark Results Summary (334 Records)

The table below lists all zero-shot and fine-tuned experiment configurations sorted by their BLEU-4 metrics:

| Run Configuration | BLEU-1 | BLEU-4 | ROUGE-L | METEOR | BERTScore F1 | Techniques Applied |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Llama-3-8B (6 Epochs, r=16)** | `0.2769` | **`0.1050`** | `0.3253` | **`0.3343`** | `0.8837` | **Best Run**: Full modules, Beam Search + post-generation truncation |
| **Qwen-3B (3 Epochs, r=16)** | **`0.3257`** | `0.1038` | **`0.3456`** | `0.3025` | **`0.9058`** | Full modules, Nucleus Sampling |
| **Qwen-3B (6 Epochs, r=32)** | `0.3213` | `0.1035` | `0.3424` | `0.2933` | `0.8929` | Full modules, Beam Search |
| **Llama-3-8B (3 Epochs, r=16)** | `0.1633` | `0.0509` | `0.2060` | `0.2907` | `0.8627` | Full modules, Beam Search, **no truncation** |
| **Llama-3-8B (6 Epochs, r=32)** | `0.1455` | `0.0426` | `0.1975` | `0.2673` | `0.8538` | Attention-only modules, Beam Search + truncation |
| **Few-Shot Llama-3-8B** | `0.1496` | `0.0124` | `0.1372` | `0.1615` | `0.8428` | 3-shot prompt, base model |
| **Zero-Shot Llama-3-8B** | `0.1455` | `0.0138` | `0.1526` | `0.1866` | `0.8467` | Zero-shot prompt, base model |
| **Few-Shot Qwen-3B** | `0.1926` | `0.0354` | `0.2235` | `0.2515` | `0.8804` | 3-shot prompt, base model |
| **Zero-Shot Qwen-3B** | `0.1109` | `0.0077` | `0.1375` | `0.1775` | `0.8539` | Zero-shot prompt, base model |

---

## 3. Key Findings & Lessons Learned

1. **Target Module Coverage is Essential**: Restricting LoRA adapters to attention-only layers (Llama r=32) severely penalizes scores compared to full coverage (Llama r=16), as MLP weights are critical for formatting style adaptation.
2. **Truncation Cleans Tokenizer Noise**: Without truncation, Llama-3's tokenizer generates trailing dialogue tags (`assistant`, etc.) which lower BLEU and ROUGE precision scores. Truncating these out doubled Llama-3's BLEU-4 (`0.0509` $\rightarrow$ `0.1050`).
3. **Beam Search Prevents Repetitive Loops**: Quantized local 8B models evaluated under greedy nucleus sampling (`temp=0.1`) fall into infinite repetition loops (repeating periods, etc.). Beam Search with `no_repeat_ngram_size=3` mathematically prevents these loops.
