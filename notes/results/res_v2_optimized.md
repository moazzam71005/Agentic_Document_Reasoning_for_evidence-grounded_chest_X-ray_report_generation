# Version 2 Optimized Metrics

Link back to: [[notes/results/results-moc.md|Results MOC]] | [[notes/versions/v2_optimized_lora_and_beam_search.md|Version 2: Optimized Pipeline]]

## Benchmark Setup
- **Evaluation Size**: 334 records (10% validation split of Indiana University dataset)
- **Model**: Qwen2.5-3B-Instruct + LoRA ($r=32$, $\alpha=64$, cosine LR decay, 6 epochs)
- **Inference Configuration**: Beam Search ($beams=4$), min length 40, token normalization

## Results Summary Table

| Metric | Zero-Shot (V2) | Few-Shot (V2) | LoRA Fine-Tuned (V2) |
| :--- | :--- | :--- | :--- |
| **BLEU-1** | `0.1263` | `0.1059` | `0.3213` |
| **BLEU-2** | `0.0521` | `0.0414` | `0.2025` |
| **BLEU-3** | `0.0247` | `0.0181` | `0.1412` |
| **BLEU-4** | `0.0116` | `0.0078` | `0.1035` |
| **ROUGE-L** | `0.1475` | `0.1132` | `0.3424` |
| **METEOR** | `0.1834` | `0.1199` | `0.2933` |
| **BERTScore F1** | `0.8450` | `0.8284` | `0.8929` |
