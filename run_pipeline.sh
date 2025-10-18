#!/bin/bash
# AlphaQuant Complete Pipeline
# 
# This script runs the complete quantization pipeline:
# 1. Compute Alpha-Hill values
# 2. Allocate bitwidth based on alpha
# 3. GPTQ weight quantization
# 4. Evaluate quantized model
# 5. Analyze results

set -e  # Exit on error

# Configuration
MODEL=${1:-"allenai/OLMoE-1B-7B-0924"}
DEVICE=${2:-"cuda"}
MXFP4_RATIO=${3:-0.3}

echo "=========================================="
echo "AlphaQuant Quantization Pipeline"
echo "=========================================="
echo "Model: $MODEL"
echo "Device: $DEVICE"
echo "MXFP4 Ratio: $MXFP4_RATIO"
echo "=========================================="

# Create results directory
mkdir -p results

# Step 1: Compute Alpha-Hill values
echo ""
echo "Step 1/5: Computing Alpha-Hill values..."
python 1_compute_alpha.py \
    --model "$MODEL" \
    --output results/alpha_values.csv \
    --device "$DEVICE"

# Step 2: Allocate bitwidth
echo ""
echo "Step 2/5: Allocating bitwidth based on alpha..."
python 2_allocate_bitwidth.py \
    --model "$MODEL" \
    --alpha-csv results/alpha_values.csv \
    --mxfp4-ratio $MXFP4_RATIO \
    --output configs/auto_quant_config.json

# Step 3: GPTQ quantization
echo ""
echo "Step 3/5: Running GPTQ quantization..."
python 3_gptq_quantize.py \
    --model "$MODEL" \
    --config configs/auto_quant_config.json \
    --dataset wikitext2 \
    --nsamples 128 \
    --save results/quantized_model.pt \
    --device "$DEVICE"

# Step 4: Evaluate model
echo ""
echo "Step 4/5: Evaluating quantized model..."
python 4_evaluate_model.py \
    --model "$MODEL" \
    --checkpoint results/quantized_model.pt \
    --tasks hellaswag,arc_easy,winogrande \
    --output results/eval_results.json \
    --device "$DEVICE"

# Step 5: Analyze results
echo ""
echo "Step 5/5: Analyzing results..."
python 5_analyze_results.py \
    --mode visualize \
    --alpha-csv results/alpha_values.csv \
    --output results/alpha_distribution.png

echo ""
echo "=========================================="
echo "✓ Pipeline Complete!"
echo "=========================================="
echo "Results saved in results/"
echo "- Alpha values: results/alpha_values.csv"
echo "- Quantization config: configs/auto_quant_config.json"
echo "- Quantized model: results/quantized_model.pt"
echo "- Evaluation results: results/eval_results.json"
echo "- Analysis plots: results/alpha_distribution.png"

