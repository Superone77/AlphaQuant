#!/bin/bash
# Step 5: Analyze results
#
# This script generates visualizations and analysis of the results.
#
# Usage:
#   ./run_5_analyze_results.sh <mode> <alpha_csv> <output>
#
# Modes:
#   - visualize: Visualize alpha distribution
#   - alpha_mse: Analyze alpha-MSE relationship
#   - compare: Compare evaluation results
#
# Examples:
#   ./run_5_analyze_results.sh visualize results/alpha_values.csv results/alpha_dist.png
#   ./run_5_analyze_results.sh alpha_mse results/alpha_values.csv results/alpha_mse.png

set -e  # Exit on error

# Configuration
MODE=${1:-"visualize"}
ALPHA_CSV=${2:-"results/alpha_values.csv"}
OUTPUT=${3:-"results/alpha_distribution.png"}

echo "=========================================="
echo "Step 5: Analyze Results"
echo "=========================================="
echo "Mode: $MODE"
echo "Alpha CSV: $ALPHA_CSV"
echo "Output: $OUTPUT"
echo "=========================================="

# Create results directory
mkdir -p results

# Run analysis
if [ "$MODE" = "alpha_mse" ]; then
    # Need model for alpha_mse mode
    MODEL=${4:-"allenai/OLMoE-1B-7B-0924"}
    echo "Model: $MODEL"
    python 5_analyze_results.py \
        --mode alpha_mse \
        --model "$MODEL" \
        --alpha-csv "$ALPHA_CSV" \
        --output "$OUTPUT"
else
    python 5_analyze_results.py \
        --mode "$MODE" \
        --alpha-csv "$ALPHA_CSV" \
        --output "$OUTPUT"
fi

echo ""
echo "✓ Step 5 complete!"
echo "Analysis saved to: $OUTPUT"
echo ""
echo "=========================================="
echo "✓ All steps complete!"
echo "=========================================="

