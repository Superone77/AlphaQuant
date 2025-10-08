#!/bin/bash
# Example: Analyze Llama 3.1 8B precision sensitivity
# This script demonstrates how to use the precision analysis tool

set -e

echo "=================================="
echo "Llama 3.1 8B Precision Analysis"
echo "=================================="

# Configuration
MODEL="meta-llama/Llama-3.1-8B"
OUTPUT_DIR="./results/llama31_8b_precision_analysis"
PRECISIONS="fp32,fp16,bf16"
DEVICE="cuda"  # Change to "cpu" if no GPU available
LOAD_DTYPE="bf16"
K_FRAC=0.1

echo ""
echo "Settings:"
echo "  Model: $MODEL"
echo "  Device: $DEVICE"
echo "  Precisions: $PRECISIONS"
echo "  Output: $OUTPUT_DIR"
echo ""

# Full analysis
echo "Running full model analysis..."
python scripts/analyze_precision_alpha_hill.py \
  --model "$MODEL" \
  --device "$DEVICE" \
  --load-dtype "$LOAD_DTYPE" \
  --precisions "$PRECISIONS" \
  --k-frac $K_FRAC \
  --output-dir "$OUTPUT_DIR/full" \
  --max-layers-per-plot 15 \
  --plot-format png \
  --log-level INFO

echo ""
echo "✓ Full analysis complete!"
echo ""

# Attention-only analysis
echo "Running attention-only analysis..."
python scripts/analyze_precision_alpha_hill.py \
  --model "$MODEL" \
  --device "$DEVICE" \
  --load-dtype "$LOAD_DTYPE" \
  --precisions "$PRECISIONS" \
  --filter-layers ".*(q_proj|k_proj|v_proj|o_proj).*" \
  --k-frac $K_FRAC \
  --output-dir "$OUTPUT_DIR/attention" \
  --max-layers-per-plot 20 \
  --plot-format png \
  --log-level INFO

echo ""
echo "✓ Attention analysis complete!"
echo ""

# MLP-only analysis
echo "Running MLP-only analysis..."
python scripts/analyze_precision_alpha_hill.py \
  --model "$MODEL" \
  --device "$DEVICE" \
  --load-dtype "$LOAD_DTYPE" \
  --precisions "$PRECISIONS" \
  --filter-layers ".*(up_proj|gate_proj|down_proj).*" \
  --k-frac $K_FRAC \
  --output-dir "$OUTPUT_DIR/mlp" \
  --max-layers-per-plot 20 \
  --plot-format png \
  --log-level INFO

echo ""
echo "✓ MLP analysis complete!"
echo ""

echo "=================================="
echo "All analyses complete!"
echo "=================================="
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Directory structure:"
echo "  $OUTPUT_DIR/"
echo "    ├── full/                      # Full model analysis"
echo "    │   ├── precision_alpha_hill_results.csv"
echo "    │   ├── precision_summary_statistics.csv"
echo "    │   └── plots/"
echo "    ├── attention/                 # Attention layers only"
echo "    │   ├── precision_alpha_hill_results.csv"
echo "    │   └── plots/"
echo "    └── mlp/                       # MLP layers only"
echo "        ├── precision_alpha_hill_results.csv"
echo "        └── plots/"
echo ""
echo "Next steps:"
echo "  1. View CSV results for numerical data"
echo "  2. Check plots/ directories for visualizations"
echo "  3. Compare summary_statistics.csv files across analyses"
echo ""

