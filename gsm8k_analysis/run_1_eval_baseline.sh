#!/bin/bash
# Step 1: Evaluate baseline OLMoE on GSM8K
#
# Usage:
#   ./gsm8k_analysis/run_1_eval_baseline.sh [model] [num_samples] [device]
#
# Examples:
#   ./gsm8k_analysis/run_1_eval_baseline.sh
#   ./gsm8k_analysis/run_1_eval_baseline.sh allenai/OLMoE-1B-7B-0924 256 cuda

set -e  # Exit on error

# Configuration
MODEL=${1:-"allenai/OLMoE-1B-7B-0924"}
NUM_SAMPLES=${2:-256}
DEVICE=${3:-"cuda"}
DTYPE=${4:-"bfloat16"}
BATCH_SIZE=${5:-1}
SEED=${6:-42}

echo "=========================================="
echo "Step 1: Baseline Evaluation"
echo "=========================================="
echo "Model: $MODEL"
echo "Samples: $NUM_SAMPLES (FIRST $NUM_SAMPLES from test set)"
echo "Random seed: $SEED"
echo "Device: $DEVICE"
echo "Dtype: $DTYPE"
echo "Batch Size: $BATCH_SIZE"
echo "=========================================="
echo ""

# Create results directory
mkdir -p gsm8k_analysis/results

# Run evaluation
python gsm8k_analysis/1_eval_baseline.py \
    --model "$MODEL" \
    --num_samples $NUM_SAMPLES \
    --batch_size $BATCH_SIZE \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    --seed $SEED \
    --output gsm8k_analysis/results/baseline.json

echo ""
echo "✓ Step 1 complete!"
echo "Results saved to: gsm8k_analysis/results/baseline.json"
echo ""
echo "Next: Run ./gsm8k_analysis/run_2_quantize_mxfp4_rtn.sh"

