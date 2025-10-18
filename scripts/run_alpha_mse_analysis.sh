#!/bin/bash
#
# Run Alpha-Hill vs MSE analysis for Llama 3.1 8B
#

set -e

# Configuration
MODEL="meta-llama/Meta-Llama-3.1-8B"
DEVICE="cuda"  # Change to "cpu" if no GPU available
DTYPE="bf16"   # bf16 for faster loading, will analyze in float32
OUTPUT_DIR="./results/alpha_mse_analysis_llama31_8b"

# Quantization formats to test
QUANT_FORMATS="mxfp8,mxfp4,fp8,fp4,int8,int6,int4,int3,int2"

# Alpha-Hill parameters
K_FRAC=0.1

# Optional: Filter specific layers (e.g., only attention layers)
# FILTER_LAYERS="attn"

# Optional: Limit number of layers for quick testing
# MAX_LAYERS=20

echo "======================================================================"
echo "Alpha-Hill vs MSE Analysis"
echo "======================================================================"
echo "Model: $MODEL"
echo "Device: $DEVICE"
echo "Output: $OUTPUT_DIR"
echo "======================================================================"
echo ""

# Build command
CMD="python scripts/analyze_alpha_mse_relationship.py \
    --model $MODEL \
    --device $DEVICE \
    --dtype $DTYPE \
    --quant-formats $QUANT_FORMATS \
    --output-dir $OUTPUT_DIR \
    --k-frac $K_FRAC \
    --log-level INFO"

# Add optional parameters if set
if [ ! -z "$FILTER_LAYERS" ]; then
    CMD="$CMD --filter-layers $FILTER_LAYERS"
fi

if [ ! -z "$MAX_LAYERS" ]; then
    CMD="$CMD --max-layers $MAX_LAYERS"
fi

echo "Running command:"
echo "$CMD"
echo ""

# Execute
eval $CMD

echo ""
echo "======================================================================"
echo "Analysis complete!"
echo "======================================================================"
echo "Results saved to: $OUTPUT_DIR"
echo "  - alpha_mse_results.csv: All data with layer names"
echo "  - alpha_mse_scatter.png: Main scatter plot"
echo "  - alpha_mse_by_category.png: Scatter plots by layer category"
echo "======================================================================"
