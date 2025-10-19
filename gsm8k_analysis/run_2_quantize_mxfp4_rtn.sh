#!/bin/bash
# Step 2: Quantize all experts to MXFP4 (RTN - no calibration)
#
# Usage:
#   ./gsm8k_analysis/run_2_quantize_mxfp4_rtn.sh [model] [num_samples] [device]
#
# Examples:
#   ./gsm8k_analysis/run_2_quantize_mxfp4_rtn.sh
#   ./gsm8k_analysis/run_2_quantize_mxfp4_rtn.sh allenai/OLMoE-1B-7B-0924 256 cuda

set -e  # Exit on error

# Configuration
MODEL=${1:-"allenai/OLMoE-1B-7B-0924"}
NUM_SAMPLES=${2:-256}
DEVICE=${3:-"cuda"}
DTYPE=${4:-"bfloat16"}
BATCH_SIZE=${5:-1}

echo "=========================================="
echo "Step 2: MXFP4 (RTN) Quantization"
echo "=========================================="
echo "Model: $MODEL"
echo "Method: RTN (no calibration)"
echo "Samples: $NUM_SAMPLES"
echo "Device: $DEVICE"
echo "Dtype: $DTYPE"
echo "Batch Size: $BATCH_SIZE"
echo "=========================================="
echo ""

# Create directories
mkdir -p gsm8k_analysis/results
mkdir -p gsm8k_analysis/configs

# Run quantization and evaluation
python gsm8k_analysis/2_quantize_mxfp4_rtn.py \
    --model "$MODEL" \
    --num_samples $NUM_SAMPLES \
    --batch_size $BATCH_SIZE \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    --output gsm8k_analysis/results/mxfp4_rtn.json \
    --save_plan gsm8k_analysis/configs/mxfp4_rtn_plan.json

echo ""
echo "✓ Step 2 complete!"
echo "Results saved to: gsm8k_analysis/results/mxfp4_rtn.json"
echo "Plan saved to: gsm8k_analysis/configs/mxfp4_rtn_plan.json"
echo ""
echo "Next: Run ./gsm8k_analysis/run_3_gptq_quantize_wikitext2.sh"

