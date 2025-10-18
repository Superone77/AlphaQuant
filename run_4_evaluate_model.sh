#!/bin/bash
# Step 4: Evaluate quantized model
#
# This script evaluates the quantized model on downstream tasks.
#
# Usage:
#   ./run_4_evaluate_model.sh <model> <device> <tasks> <batch_size>
#
# Examples:
#   ./run_4_evaluate_model.sh allenai/OLMoE-1B-7B-0924 cuda "hellaswag,arc_easy" 8
#   ./run_4_evaluate_model.sh meta-llama/Llama-2-7b-hf cuda "mmlu,gsm8k" 4

set -e  # Exit on error

# Configuration
MODEL=${1:-"allenai/OLMoE-1B-7B-0924"}
DEVICE=${2:-"cuda"}
TASKS=${3:-"hellaswag,arc_easy,winogrande"}
BATCH_SIZE=${4:-8}
CHECKPOINT=${5:-"results/quantized_model.pt"}
OUTPUT=${6:-"results/eval_results.json"}

echo "=========================================="
echo "Step 4: Evaluate Model"
echo "=========================================="
echo "Model: $MODEL"
echo "Device: $DEVICE"
echo "Checkpoint: $CHECKPOINT"
echo "Tasks: $TASKS"
echo "Batch Size: $BATCH_SIZE"
echo "Output: $OUTPUT"
echo "=========================================="

# Create results directory
mkdir -p results

# Run evaluation
python 4_evaluate_model.py \
    --model "$MODEL" \
    --checkpoint "$CHECKPOINT" \
    --tasks "$TASKS" \
    --batch_size $BATCH_SIZE \
    --output "$OUTPUT" \
    --device "$DEVICE"

echo ""
echo "✓ Step 4 complete!"
echo "Evaluation results saved to: $OUTPUT"
echo ""
echo "Next step: Run ./run_5_analyze_results.sh"

