#!/bin/bash
# Step 2.1: Uniform Bitwidth Allocation for Experts
#
# This script creates a quantization config with uniform allocation:
# - Front half experts: one precision level
# - Back half experts: another precision level
#
# Usage:
#   ./run_2.1_uniform_allocate_bitwidth.sh <model> <front_precision> <back_precision>
#
# Examples:
#   # Mixed precision: front half mxfp6, back half mxfp4
#   ./run_2.1_uniform_allocate_bitwidth.sh allenai/OLMoE-1B-7B-0924 mxfp6 mxfp4
#   
#   # Integer quantization: front half int4, back half int3
#   ./run_2.1_uniform_allocate_bitwidth.sh allenai/OLMoE-1B-7B-0924 int4 int3
#   
#   # Conservative: front half mxfp8, back half mxfp6
#   ./run_2.1_uniform_allocate_bitwidth.sh meta-llama/Llama-2-7b-hf mxfp8 mxfp6

set -e  # Exit on error

# Configuration
MODEL=${1:-"allenai/OLMoE-1B-7B-0125-Instruct"}
FRONT_PRECISION=${2:-"mxfp6"}
BACK_PRECISION=${3:-"mxfp4"}
FRONT_GROUP_SIZE=${4:-128}
BACK_GROUP_SIZE=${5:-128}
DEFAULT_PRECISION=${6:-"mxfp8"}
DEFAULT_GROUP_SIZE=${7:-128}
OUTPUT=${8:-"configs/uniform_quant_config.json"}

echo "=========================================="
echo "Step 2.1: Uniform Bitwidth Allocation"
echo "=========================================="
echo "Model: $MODEL"
echo "Front Half Precision: $FRONT_PRECISION (group_size=$FRONT_GROUP_SIZE)"
echo "Back Half Precision: $BACK_PRECISION (group_size=$BACK_GROUP_SIZE)"
echo "Default Precision: $DEFAULT_PRECISION (group_size=$DEFAULT_GROUP_SIZE)"
echo "Output: $OUTPUT"
echo "=========================================="

# Create configs directory
mkdir -p configs

# Run uniform bitwidth allocation
python 2.1_uniform_allocate_bitwidth.py \
    --model "$MODEL" \
    --front-precision $FRONT_PRECISION \
    --back-precision $BACK_PRECISION \
    --front-group-size $FRONT_GROUP_SIZE \
    --back-group-size $BACK_GROUP_SIZE \
    --default-precision $DEFAULT_PRECISION \
    --default-group-size $DEFAULT_GROUP_SIZE \
    --output "$OUTPUT"

echo ""
echo "✓ Step 2.1 complete!"
echo "Quantization config saved to: $OUTPUT"
echo ""
echo "Next step: Run ./run_3_gptq_quantize.sh with the generated config"

