# Version 3: Llama-3-OpenBioLLM-8B Remote Pipeline

Documents the remote QLoRA training and evaluation parameters for Llama-3-OpenBioLLM-8B on the Ubuntu PC.

Link back to: [[notes/versions/versions-moc.md|Versions MOC]] | [[notes/context/overview.md|Project Overview]]

## Model & Quantization Configuration
* **Base Model**: `aaditya/Llama3-OpenBioLLM-8B` (loaded on remote GPU in NF4 4-bit quantization, with unquantized embedding weights forced to `bfloat16` to prevent memory bottlenecks).
* **LoRA Configuration**: LoRA adapters with rank $r=16$ / scaling factor $\alpha=32$ targeted at all linear projection layers to fit within the 10GB RTX 3080 VRAM budget.

## Training Configuration
* **Hyperparameters**: 3 epochs, learning rate `1e-4`, cosine schedule, warmup ratio `0.05`, batch size 1 with gradient accumulation 16 (effective batch size 16).
* **VRAM Optimizations**: Enabled gradient checkpointing, `expandable_segments=True` PyTorch memory allocator configuration, and reduced max sequence length to `256` tokens.

## Associated Links
* **Results**: [[notes/results/res_v3_llama.md|Llama-3-OpenBioLLM-8B Results]]
* **Scripts**: [train_llama.py](file:///c:/Users/Taimoor/Desktop/FYP/ADRE-Xray/Agentic_Document_Reasoning_for_evidence-grounded_chest_X-ray_report_generation/adre-cxr/scripts/train_llama.py) & [evaluate_full_testset_llama.py](file:///c:/Users/Taimoor/Desktop/FYP/ADRE-Xray/Agentic_Document_Reasoning_for_evidence-grounded_chest_X-ray_report_generation/adre-cxr/scripts/evaluate_full_testset_llama.py)
