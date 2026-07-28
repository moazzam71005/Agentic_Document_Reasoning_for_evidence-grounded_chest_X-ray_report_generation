#!/bin/bash
# Master pipeline script to fine-tune Llama3-OpenBioLLM-8B and run evaluations sequentially.

set -e  # Exit on error

# Redirect all stdout and stderr to logging file
LOG_FILE="/home/tukl/Taimoor/ADRE-CXR/pipeline_llama.log"
exec > >(tee -i "$LOG_FILE") 2>&1

echo "=================================================="
echo "Starting Remote Llama-3-OpenBioLLM-8B Pipeline"
echo "Start Time: $(date)"
echo "=================================================="

PYTHON_BIN="/home/tukl/Taimoor/miniconda_envs/adre_cxr/bin/python"
PROJECT_DIR="/home/tukl/Taimoor/ADRE-CXR"

echo "--------------------------------------------------"
echo "1. Running Supervised Fine-Tuning (SFT)..."
echo "--------------------------------------------------"
"$PYTHON_BIN" "$PROJECT_DIR/scripts/train_llama.py" --epochs 3

echo "--------------------------------------------------"
echo "2. Running Zero-Shot, Few-Shot, and LoRA FT Evaluation..."
echo "--------------------------------------------------"
"$PYTHON_BIN" "$PROJECT_DIR/scripts/evaluate_full_testset_llama.py"

echo "=================================================="
echo "Pipeline Completed Successfully!"
echo "End Time: $(date)"
echo "=================================================="
