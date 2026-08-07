#!/usr/bin/env bash
set -e
echo "============================================================"
echo "Phase 5: Training Llama on Pipeline SFT Data (from Qwen)"
echo "============================================================"
PYTHONPATH=. /home/tukl/Taimoor/miniconda_envs/adre_cxr/bin/python scripts/train_pipeline.py \
    --model-id /data/Taimoor/ADRE-CXR/weights/Llama3-OpenBioLLM-8B \
    --sft-data output/pipeline_sft_data_qwen.json \
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
