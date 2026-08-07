#!/usr/bin/env bash

# run_full_sft_pipeline.sh
# Orchestrates SFT generation, training, and evaluation for Qwen and Llama

set -e

# Phase 1 and 2 completed successfully. Resuming from Phase 3.

echo "============================================================"
echo "Phase 3: Evaluating Fine-Tuned Qwen Pipeline"
echo "============================================================"
PYTHONPATH=. /home/tukl/Taimoor/miniconda_envs/adre_cxr/bin/python scripts/evaluate_full_pipeline.py \
    --model-path /home/tukl/Taimoor/ADRE-CXR/weights/qwen2.5-3b-pipeline-ft-merged \
    --cache-name cache_pipeline_qwen_ft \
    --output-json output/full_pipeline_metrics_qwen_ft.json

echo "============================================================"
echo "Phase 4: Generating Pipeline SFT Data for Llama"
echo "============================================================"
PYTHONPATH=. /home/tukl/Taimoor/miniconda_envs/adre_cxr/bin/python scripts/generate_pipeline_sft_data.py \
    --model-path /data/Taimoor/ADRE-CXR/weights/Llama-3-OpenBioLLM-8B \
    --output-json output/pipeline_sft_data_llama.json

echo "============================================================"
echo "Phase 5: Training Llama on Pipeline SFT Data"
echo "============================================================"
PYTHONPATH=. /home/tukl/Taimoor/miniconda_envs/adre_cxr/bin/python scripts/train_pipeline.py \
    --model-id /data/Taimoor/ADRE-CXR/weights/Llama-3-OpenBioLLM-8B \
    --sft-data output/pipeline_sft_data_llama.json \
    --output-dir weights/llama-3-8b-pipeline-ft

echo "============================================================"
echo "Phase 6: Evaluating Fine-Tuned Llama Pipeline"
echo "============================================================"
PYTHONPATH=. /home/tukl/Taimoor/miniconda_envs/adre_cxr/bin/python scripts/evaluate_full_pipeline.py \
    --model-path /home/tukl/Taimoor/ADRE-CXR/weights/llama-3-8b-pipeline-ft-merged \
    --cache-name cache_pipeline_llama_ft_post \
    --output-json output/full_pipeline_metrics_llama_ft_post.json

echo "============================================================"
echo "ALL PIPELINES COMPLETED SUCCESSFULLY!"
echo "============================================================"
