#!/bin/bash
# Step 4: GPTQ + MXFP4 quantization with GSM8K calibration
#
# Usage:
#   ./gsm8k_analysis/run_4_gptq_quantize_gsm8k.sh [model] [calib_samples] [eval_samples] [device]
#
# Examples:
#   ./gsm8k_analysis/run_4_gptq_quantize_gsm8k.sh
#   ./gsm8k_analysis/run_4_gptq_quantize_gsm8k.sh allenai/OLMoE-1B-7B-0924 128 256 cuda

set -e  # Exit on error

# Configuration
MODEL=${1:-"allenai/OLMoE-1B-7B-0924"}
NUM_CALIB_SAMPLES=${2:-128}
NUM_EVAL_SAMPLES=${3:-256}
DEVICE=${4:-"cuda"}
DTYPE=${5:-"bfloat16"}
BATCH_SIZE=${6:-1}

echo "=========================================="
echo "Step 4: GPTQ + MXFP4 (GSM8K)"
echo "=========================================="
echo "Model: $MODEL"
echo "Calibration: GSM8K (task-specific)"
echo "Calibration samples: $NUM_CALIB_SAMPLES"
echo "Eval samples: $NUM_EVAL_SAMPLES"
echo "Device: $DEVICE"
echo "Dtype: $DTYPE"
echo "Batch Size: $BATCH_SIZE"
echo "=========================================="
echo ""

# Create directories
mkdir -p gsm8k_analysis/results
mkdir -p gsm8k_analysis/configs

# Run GPTQ quantization and evaluation
python gsm8k_analysis/4_gptq_quantize_gsm8k.py \
    --model "$MODEL" \
    --num_calibration_samples $NUM_CALIB_SAMPLES \
    --num_eval_samples $NUM_EVAL_SAMPLES \
    --batch_size $BATCH_SIZE \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    --output gsm8k_analysis/results/gptq_mxfp4_gsm8k.json \
    --save_plan gsm8k_analysis/configs/gptq_mxfp4_gsm8k_plan.json

echo ""
echo "✓ Step 4 complete!"
echo "Results saved to: gsm8k_analysis/results/gptq_mxfp4_gsm8k.json"
echo "Plan saved to: gsm8k_analysis/configs/gptq_mxfp4_gsm8k_plan.json"
echo ""
echo "Next: Run ./gsm8k_analysis/run_5_analyze_results.sh"

