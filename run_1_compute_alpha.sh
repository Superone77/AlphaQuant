#!/bin/bash
# Step 1: Compute Alpha-Hill values
#
# This script computes Alpha-Hill sensitivity values for each layer.
#
# Usage:
#   ./run_1_compute_alpha.sh <model> <device>
#
# Examples:
#   ./run_1_compute_alpha.sh allenai/OLMoE-1B-7B-0924 cuda
#   ./run_1_compute_alpha.sh meta-llama/Llama-2-7b-hf cpu

set -e  # Exit on error

# Configuration
MODEL=${1:-"allenai/OLMoE-1B-7B-0924"}
DEVICE=${2:-"cuda"}
OUTPUT=${3:-"results/alpha_values.csv"}

echo "=========================================="
echo "Step 1: Compute Alpha-Hill Values"
echo "=========================================="
echo "Model: $MODEL"
echo "Device: $DEVICE"
echo "Output: $OUTPUT"
echo "=========================================="

# Create results directory
mkdir -p results

# Run alpha computation
python 1_compute_alpha.py \
    --model "$MODEL" \
    --output "$OUTPUT" \
    --device "$DEVICE"

echo ""
echo "✓ Step 1 complete!"
echo "Alpha values saved to: $OUTPUT"
echo ""
echo "Next step: Run ./run_2_allocate_bitwidth.sh"

