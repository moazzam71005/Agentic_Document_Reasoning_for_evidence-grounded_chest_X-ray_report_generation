# Version 1 Baseline Metrics

Link back to: [[notes/results/results-moc.md|Results MOC]] | [[notes/versions/v1_baseline.md|Version 1: Baseline Pipeline]]

## Benchmark Setup
- **Evaluation Size**: 334 records (10% validation split of Indiana University dataset)
- **Baseline Model**: Qwen2.5-3B-Instruct

## Results Summary Table

| Metric | Zero-Shot | Few-Shot (3-Shot) | LoRA Fine-Tuned ($r=16$) |
| :--- | :--- | :--- | :--- |
| **BLEU-1** | `0.1109` | `0.1926` | `0.3257` |
| **BLEU-2** | `0.0407` | `0.1021` | `0.2028` |
| **BLEU-3** | `0.0173` | `0.0604` | `0.1419` |
| **BLEU-4** | `0.0077` | `0.0354` | `0.1038` |
| **ROUGE-L** | `0.1375` | `0.2235` | `0.3456` |
| **METEOR** | `0.1775` | `0.2515` | `0.3025` |
| **BERTScore F1** | `0.8539` | `0.8804` | `0.9058` |
