# Qwen2.5-3B SFT Fine-Tuning Training Configuration

Link back to: [[notes/context/overview.md|Project Overview]]

## Context & Constraints
Fine-tuning a 3.09 billion parameter model (`Qwen2.5-3B-Instruct`) locally requires substantial VRAM. The system operates on a GeForce RTX 3050 laptop GPU with a hardware constraint of **6 GB VRAM**. A standard 16-bit or 8-bit training configuration would instantly trigger Out of Memory (OOM) errors.

## Core Decisions
* **QLoRA (4-bit Quantization)**: Load the base model in 4-bit precision via bitsandbytes, freeze the base model parameters, and attach 16-rank LoRA adapters to all linear projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
* **Active Resuming**: Auto-scan the checkpoints directory to immediately resume training from the latest checkpoint (`checkpoint-XXX`) if a system crash or power loss interrupts execution.

## Optimization Parameters
To stay below 6 GB of VRAM, the training options are customized:
1. **Batch Size Tuning**: Physical batch size is set to `1` sample, with gradient accumulation set to `16` to maintain a stable effective training batch size of 16.
2. **Gradient Checkpointing**: Enabled to compute activations on-the-fly during backpropagation instead of caching them in memory.
3. **Paged Optimizer**: Used `paged_adamw_8bit` optimizer to offload optimizer state pages to CPU memory when VRAM is full.
