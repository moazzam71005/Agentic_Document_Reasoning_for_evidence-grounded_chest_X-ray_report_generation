# Subset Comparative Evaluation (26 Records)

Link back to: [[notes/experiments/experiments-moc.md|Experiments MOC]] | [[notes/context/overview.md|Project Overview]]

## Experiment Setup
- **Model**: `Qwen/Qwen2.5-3B-Instruct`
- **Precision**: 4-bit NormalFloat (`nf4`) loading using `bitsandbytes`
- **Sample Size**: 26 custom-selected validation split record UIDs (designed as a fast validation subset)
- **Execution Script**: `adre-cxr/scratch/run_evaluation_comparison.py`

## Comparative Metrics
The model was tested across three configurations:
1. **Zero-Shot**: Direct instruction-following prompt.
2. **Few-Shot**: Direct instruction-following with 3 in-context examples.
3. **LoRA Fine-Tuned (FT)**: 4-bit LoRA adapter loaded over base model.

| Metric | Zero-Shot | Few-Shot (3-Shot) | LoRA Fine-Tuned |
| :--- | :--- | :--- | :--- |
| **BLEU-1** | `0.1086` | `0.2039` | `0.3665` |
| **BLEU-4** | `0.0078` | `0.0407` | `0.1025` |
| **ROUGE-L** | `0.1490` | `0.2465` | `0.3681` |
| **BERTScore F1** | `0.8559` | `0.8858` | `0.9131` |

## Key Findings
- **Vocabulary Alignment**: Zero-shot generated correct diagnostic reasoning but struggled with vocabulary and stylistic conventions of radiology reports.
- **Few-Shot Efficacy**: 3-shot examples helped align the base model style, doubling the BLEU-1 score and increasing ROUGE-L significantly.
- **LoRA Dominance**: LoRA Fine-Tuning achieved the best score by far, producing highly precise and structured reports.
