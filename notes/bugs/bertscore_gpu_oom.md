# BERTScore GPU Out-of-Memory Bug

During SFT evaluation of the Llama-3-OpenBioLLM-8B model, loading RoBERTa-large on the GPU simultaneously with the 8B model caused a CUDA Out-of-Memory (OOM) crash on the 10GB RTX 3080.

Link back to: [[notes/context/overview.md|Overview]] | [[notes/bugs/bugs-moc.md|Bugs MOC]]

## Problem
Llama-3-OpenBioLLM-8B takes ~8.3 GB of VRAM during quantized loading. When the baseline evaluation function calls `bert_score` with `roberta-large` (which occupies ~1.4 GB VRAM), the combined footprint exceeded the physical 10GB GPU memory capacity, leading to a crash.

## Solution
Modified the evaluation helper in `metrics.py` and `evaluate_full_testset_llama.py` to run BERTScore calculations on the CPU by passing `device="cpu"`. 

## Outcome
Running BERTScore on the CPU successfully resolved the GPU memory conflicts and allowed the SFT pipeline to run stably, with only ~20 seconds of calculation overhead per pass.
