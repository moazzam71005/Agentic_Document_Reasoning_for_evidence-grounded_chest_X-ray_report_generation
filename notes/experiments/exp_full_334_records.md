# Full Validation Split Comparative Evaluation (334 Records)

Link back to: [[notes/experiments/experiments-moc.md|Experiments MOC]] | [[notes/context/overview.md|Project Overview]]

## Experiment Setup
- **Model**: `Qwen/Qwen2.5-3B-Instruct`
- **Precision**: 4-bit NormalFloat (`nf4`) loading using `bitsandbytes`
- **Sample Size**: 334 records (the complete 10% validation split of the 3,331 Indiana University chest X-ray reports dataset)
- **Execution Script**: `adre-cxr/scripts/evaluate_full_testset.py`
- **Robustness**: Utilized incremental JSON caching to write predictions to disk after each sample, allowing pause/resume functionality.

## Comparative Metrics

### Version 1: Baseline (Nucleus Sampling, LoRA $r=16$)
* **Inference**: Nucleus sampling (`temperature=0.1`, `do_sample=True`)
* **Text Processing**: Raw text output (no normalization)

| Metric | Zero-Shot | Few-Shot (3-Shot) | LoRA Fine-Tuned |
| :--- | :--- | :--- | :--- |
| **BLEU-1** | `0.1109` | `0.1926` | `0.3257` |
| **BLEU-2** | `0.0407` | `0.1021` | `0.2028` |
| **BLEU-3** | `0.0173` | `0.0604` | `0.1419` |
| **BLEU-4** | `0.0077` | `0.0354` | `0.1038` |
| **ROUGE-L** | `0.1375` | `0.2235` | `0.3456` |
| **METEOR** | `0.1775` | `0.2515` | `0.3025` |
| **BERTScore F1** | `0.8539` | `0.8804` | `0.9058` |

### Version 2: Optimized (Beam Search, LoRA $r=32$ + Normalization)
* **Inference**: Beam Search (`num_beams=4`, `min_new_tokens=40`, `length_penalty=1.0`)
* **Text Processing**: Standardized token normalization (`normalize_report` converts lowercase, normalizes bullet lists)

| Metric | Zero-Shot | Few-Shot (3-Shot) | LoRA Fine-Tuned |
| :--- | :--- | :--- | :--- |
| **BLEU-1** | `0.1263` | `0.1059` | `0.3213` |
| **BLEU-2** | `0.0521` | `0.0414` | `0.2025` |
| **BLEU-3** | `0.0247` | `0.0181` | `0.1412` |
| **BLEU-4** | `0.0116` | `0.0078` | `0.1035` |
| **ROUGE-L** | `0.1475` | `0.1132` | `0.3424` |
| **METEOR** | `0.1834` | `0.1199` | `0.2933` |
| **BERTScore F1** | `0.8450` | `0.8284` | `0.8929` |

---

## Key Findings & Version Comparison
- **Zero-Shot Boost**: Beam search and text normalization improved Zero-Shot BLEU-4 by **50.6%** (from `0.0077` to `0.0116`) and ROUGE-L by **7.3%** (from `0.1375` to `0.1475`).
- **Few-Shot Degradation**: In Few-Shot mode, combining the long multi-shot prompt context with Beam Search constraints caused a significant drop in BLEU/ROUGE scores. The model tended to over-rely on repeating template fragments or generating static text patterns under search constraints.
- **LoRA FT Generalization Boundary**: Despite doubling the LoRA capacity to $r=32$ and training for 6 epochs (where local validation set BLEU-4 reached **`0.1626`**), the performance on the out-of-overlap test split remained virtually identical to Version 1 (`0.1035` vs. `0.1038`). This suggests the model has reached a generalization ceiling on the Indiana University dataset, and further capacity or training epochs leads to style overfitting.
