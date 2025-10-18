#!/bin/bash
# Step 3: GPTQ weight quantization
#
# This script applies GPTQ quantization using the config from step 2.
#
# Usage:
#   ./run_3_gptq_quantize.sh <model> <device> <dataset> <nsamples>
#
# Examples:
#   ./run_3_gptq_quantize.sh allenai/OLMoE-1B-7B-0924 cuda wikitext2 128
#   ./run_3_gptq_quantize.sh meta-llama/Llama-2-7b-hf cuda c4 256

set -e  # Exit on error

# Configuration
MODEL=${1:-"allenai/OLMoE-1B-7B-0924"}
DEVICE=${2:-"cuda"}
DATASET=${3:-"wikitext2"}
NSAMPLES=${4:-128}
CONFIG=${5:-"configs/auto_quant_config.json"}
OUTPUT=${6:-"results/quantized_model.pt"}

echo "=========================================="
echo "Step 3: GPTQ Quantization"
echo "=========================================="
echo "Model: $MODEL"
echo "Device: $DEVICE"
echo "Config: $CONFIG"
echo "Dataset: $DATASET"
echo "Samples: $NSAMPLES"
echo "Output: $OUTPUT"
echo "=========================================="

# Create results directory
mkdir -p results

# Run GPTQ quantization
python 3_gptq_quantize.py \
    --model "$MODEL" \
    --config "$CONFIG" \
    --dataset "$DATASET" \
    --nsamples $NSAMPLES \
    --save "$OUTPUT" \
    --device "$DEVICE"

echo ""
echo "✓ Step 3 complete!"
echo "Quantized model saved to: $OUTPUT"
echo ""
echo "Next step: Run ./run_4_evaluate_model.sh"

