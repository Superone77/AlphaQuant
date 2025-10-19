#!/bin/bash
# GSM8K Analysis Pipeline Runner
#
# This script runs the complete GSM8K analysis pipeline for OLMoE quantization.
#
# Usage:
#   ./gsm8k_analysis/run_pipeline.sh [model] [num_samples] [device]
#
# Examples:
#   ./gsm8k_analysis/run_pipeline.sh
#   ./gsm8k_analysis/run_pipeline.sh allenai/OLMoE-1B-7B-0924 256 cuda
#   ./gsm8k_analysis/run_pipeline.sh allenai/OLMoE-1B-7B-0924 128 cuda:0

set -e  # Exit on error

# Configuration
MODEL=${1:-"allenai/OLMoE-1B-7B-0924"}
NUM_SAMPLES=${2:-256}
NUM_CALIBRATION_SAMPLES=${3:-128}
DEVICE=${4:-"cuda"}
DTYPE=${5:-"bfloat16"}
BATCH_SIZE=${6:-1}

echo "=========================================="
echo "GSM8K Analysis Pipeline"
echo "=========================================="
echo "Model: $MODEL"
echo "GSM8K Samples: $NUM_SAMPLES"
echo "Calibration Samples: $NUM_CALIBRATION_SAMPLES"
echo "Device: $DEVICE"
echo "Dtype: $DTYPE"
echo "Batch Size: $BATCH_SIZE"
echo "=========================================="
echo ""

# Create results directory
mkdir -p gsm8k_analysis/results
mkdir -p gsm8k_analysis/configs

# Run pipeline
python gsm8k_analysis/run_pipeline.py \
    --model "$MODEL" \
    --num_samples $NUM_SAMPLES \
    --num_calibration_samples $NUM_CALIBRATION_SAMPLES \
    --batch_size $BATCH_SIZE \
    --device "$DEVICE" \
    --dtype "$DTYPE"

echo ""
echo "=========================================="
echo "✓ Pipeline complete!"
echo "=========================================="
echo ""
echo "Results saved in: gsm8k_analysis/results/"
echo ""
echo "To view results:"
echo "  - Check CSV: gsm8k_analysis/results/comparison_*.csv"
echo "  - Check detailed analysis: gsm8k_analysis/results/comparison_*.txt"
echo ""

