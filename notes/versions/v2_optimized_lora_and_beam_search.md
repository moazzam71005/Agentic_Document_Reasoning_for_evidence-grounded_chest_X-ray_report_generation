# Version 2: Expanded LoRA Capacity & Beam Search

Link back to: [[notes/versions/versions-moc.md|Versions MOC]] | [[notes/context/overview.md|Project Overview]]

## Pipeline Configuration
- **Base Model**: `Qwen/Qwen2.5-3B-Instruct` (quantized to 4-bit NF4)
- **Fine-Tuning Architecture**: LoRA with expanded rank $r=32$ and scaling factor $\alpha=64$
- **Target Modules**: All linear projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`)
- **Training Epochs**: 6 epochs
- **Hyperparameters**: Lower learning rate `1e-4` (for stability on more epochs), cosine scheduler, warmup ratio `0.05`, gradient accumulation 16
- **Inference Strategy**: Beam Search generation:
  - `num_beams=4`
  - `early_stopping=True`
  - `no_repeat_ngram_size=3`
  - `length_penalty=1.0` (prevents early truncation)
  - `min_new_tokens=40` (neutralizes BLEU brevity penalty)
- **Text Processing**: Standardized `normalize_report` preprocessing that converts characters to lowercase, replaces bullet variants (`-`, `•` -> `*`), and consolidates whitespaces.

## Associated Links
- **Metrics Results**: [[notes/results/res_v2_optimized.md|Version 2 Optimized Results]]
- **Full Experiment**: [[notes/experiments/exp_full_334_records.md|Full Validation Split Comparative Experiment]]
