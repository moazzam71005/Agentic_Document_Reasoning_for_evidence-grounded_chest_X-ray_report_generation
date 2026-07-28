# Version 1: Baseline Prompting & SFT

Link back to: [[notes/versions/versions-moc.md|Versions MOC]] | [[notes/context/overview.md|Project Overview]]

## Pipeline Configuration
- **Base Model**: `Qwen/Qwen2.5-3B-Instruct` (quantized to 4-bit NF4)
- **Fine-Tuning Architecture**: LoRA with rank $r=16$ and scaling factor $\alpha=32$
- **Target Modules**: All linear projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`)
- **Training Epochs**: 3 epochs
- **Hyperparameters**: Learning rate `2e-4`, cosine scheduler, warmup ratio `0.03`, gradient accumulation 16
- **Inference Strategy**: Nucleus sampling with `temperature=0.1` and `do_sample=True`
- **Text Processing**: Standard decoding without text normalization or formatting alignments

## Associated Links
- **Metrics Results**: [[notes/results/res_v1_baseline.md|Version 1 Baseline Results]]
- **Full Experiment**: [[notes/experiments/exp_full_334_records.md|Full Validation Split Comparative Experiment]]
- **Subset Experiment**: [[notes/experiments/exp_subset_26_records.md|Subset Comparative Experiment]]
