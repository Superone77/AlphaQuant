#!/bin/bash
# Step 5: Analyze and compare all results
#
# Usage:
#   ./gsm8k_analysis/run_5_analyze_results.sh
#
# Examples:
#   ./gsm8k_analysis/run_5_analyze_results.sh

set -e  # Exit on error

echo "=========================================="
echo "Step 5: Analyze Results"
echo "=========================================="
echo ""

# Run analysis
python gsm8k_analysis/5_analyze_results.py \
    --baseline gsm8k_analysis/results/baseline.json \
    --mxfp4_rtn gsm8k_analysis/results/mxfp4_rtn.json \
    --gptq_wikitext2 gsm8k_analysis/results/gptq_mxfp4_wikitext2.json \
    --gptq_gsm8k gsm8k_analysis/results/gptq_mxfp4_gsm8k.json \
    --output gsm8k_analysis/results/comparison.csv

echo ""
echo "✓ Step 5 complete!"
echo "Comparison saved to: gsm8k_analysis/results/comparison.csv"
echo "Detailed analysis saved to: gsm8k_analysis/results/comparison.txt"
echo ""
echo "All steps completed! Check the results directory for outputs."

