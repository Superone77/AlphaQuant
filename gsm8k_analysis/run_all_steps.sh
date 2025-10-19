#!/bin/bash
# Run all GSM8K analysis steps sequentially
#
# This script runs the complete pipeline:
# 1. Baseline evaluation
# 2. MXFP4 (RTN) quantization
# 3. GPTQ + MXFP4 (WikiText2 calibration)
# 4. GPTQ + MXFP4 (GSM8K calibration)
# 5. Results analysis
#
# Usage:
#   ./gsm8k_analysis/run_all_steps.sh [model] [eval_samples] [calib_samples] [device]
#
# Examples:
#   ./gsm8k_analysis/run_all_steps.sh
#   ./gsm8k_analysis/run_all_steps.sh allenai/OLMoE-1B-7B-0924 256 128 cuda

set -e  # Exit on error

# Configuration
MODEL=${1:-"allenai/OLMoE-1B-7B-0924"}
NUM_EVAL_SAMPLES=${2:-256}
NUM_CALIB_SAMPLES=${3:-128}
DEVICE=${4:-"cuda"}
DTYPE=${5:-"bfloat16"}
SEED=${6:-42}

echo "=========================================="
echo "GSM8K Analysis - Complete Pipeline"
echo "=========================================="
echo "Model: $MODEL"
echo "Eval samples: $NUM_EVAL_SAMPLES (FIRST $NUM_EVAL_SAMPLES from test set)"
echo "Calibration samples: $NUM_CALIB_SAMPLES"
echo "Random seed: $SEED (for reproducibility)"
echo "Device: $DEVICE"
echo "Dtype: $DTYPE"
echo "=========================================="
echo ""
echo "Note: All experiments will evaluate on the SAME samples!"
echo "      lm_eval's 'limit' ensures deterministic sample selection."
echo "=========================================="
echo ""

# Step 1: Baseline
echo "Running Step 1/5: Baseline Evaluation..."
./gsm8k_analysis/run_1_eval_baseline.sh "$MODEL" $NUM_EVAL_SAMPLES "$DEVICE" "$DTYPE" 1 $SEED

# Step 2: MXFP4 (RTN)
echo ""
echo "Running Step 2/5: MXFP4 (RTN) Quantization..."
./gsm8k_analysis/run_2_quantize_mxfp4_rtn.sh "$MODEL" $NUM_EVAL_SAMPLES "$DEVICE" "$DTYPE" 1 $SEED

# Step 3: GPTQ + WikiText2
echo ""
echo "Running Step 3/5: GPTQ + MXFP4 (WikiText2)..."
./gsm8k_analysis/run_3_gptq_quantize_wikitext2.sh "$MODEL" $NUM_CALIB_SAMPLES $NUM_EVAL_SAMPLES "$DEVICE" "$DTYPE" 1 $SEED

# Step 4: GPTQ + GSM8K
echo ""
echo "Running Step 4/5: GPTQ + MXFP4 (GSM8K)..."
./gsm8k_analysis/run_4_gptq_quantize_gsm8k.sh "$MODEL" $NUM_CALIB_SAMPLES $NUM_EVAL_SAMPLES "$DEVICE" "$DTYPE" 1 $SEED

# Step 5: Analysis
echo ""
echo "Running Step 5/5: Results Analysis..."
./gsm8k_analysis/run_5_analyze_results.sh

echo ""
echo "=========================================="
echo "✓ All steps completed successfully!"
echo "=========================================="
echo ""
echo "Sample Logs (with detailed reasoning):"
echo "  - Baseline: gsm8k_analysis/results/baseline_samples.log"
echo "  - MXFP4 (RTN): gsm8k_analysis/results/mxfp4_rtn_samples.log"
echo "  - GPTQ + WikiText2: gsm8k_analysis/results/gptq_mxfp4_wikitext2_samples.log"
echo "  - GPTQ + GSM8K: gsm8k_analysis/results/gptq_mxfp4_gsm8k_samples.log"
echo ""
echo "Analysis:"
echo "  - Comparison: gsm8k_analysis/results/comparison.csv"
echo "  - Detailed: gsm8k_analysis/results/comparison.txt"
echo ""
echo "Verify sample consistency:"
echo "  python gsm8k_analysis/verify_consistency.py"
echo ""

