#!/bin/bash
# Step 2: Allocate bitwidth based on Alpha-Hill values
#
# This script creates a quantization config based on alpha sensitivity.
# By default, skips attention and routing gate layers.
# Mixed precision is applied to gate_proj/up_proj/down_proj.
#
# Usage:
#   ./run_2_allocate_bitwidth.sh <model> <mxfp4_ratio> <bf16_ratio>
#
# Examples:
#   ./run_2_allocate_bitwidth.sh allenai/OLMoE-1B-7B-0924 0.3 0.0
#   ./run_2_allocate_bitwidth.sh meta-llama/Llama-2-7b-hf 0.5 0.1

set -e  # Exit on error

# Configuration
MODEL=${1:-"allenai/OLMoE-1B-7B-0924"}
MXFP4_RATIO=${2:-0.3}
BF16_RATIO=${3:-0.0}
ALPHA_CSV=${4:-"results/alpha_values.csv"}
OUTPUT=${5:-"configs/auto_quant_config.json"}

echo "=========================================="
echo "Step 2: Allocate Bitwidth"
echo "=========================================="
echo "Model: $MODEL"
echo "Alpha CSV: $ALPHA_CSV"
echo "MXFP4 Ratio: $MXFP4_RATIO"
echo "BF16 Ratio: $BF16_RATIO"
echo "Output: $OUTPUT"
echo "=========================================="

# Create configs directory
mkdir -p configs

# Run bitwidth allocation
python 2_allocate_bitwidth.py \
    --model "$MODEL" \
    --alpha-csv "$ALPHA_CSV" \
    --mxfp4-ratio $MXFP4_RATIO \
    --bf16-ratio $BF16_RATIO \
    --output "$OUTPUT"

echo ""
echo "✓ Step 2 complete!"
echo "Quantization config saved to: $OUTPUT"
echo ""
echo "Next step: Run ./run_3_gptq_quantize.sh"

