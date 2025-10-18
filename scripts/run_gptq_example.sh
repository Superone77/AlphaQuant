#!/bin/bash

# Example script for running GPTQ quantization with AlphaQuant
# Usage: ./run_gptq_example.sh [model_name] [config_path]

set -e  # Exit on error

# Default values
MODEL="${1:-meta-llama/Llama-2-7b-hf}"
CONFIG="${2:-configs/gptq_mixed_precision.json}"
DATASET="wikitext2"
NSAMPLES=128
SEQLEN=2048
METHOD="gptq"
BLOCKSIZE=128
PERCDAMP=0.01

# Output path
OUTPUT_DIR="./gptq_outputs"
mkdir -p "$OUTPUT_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_FILE="${OUTPUT_DIR}/quantized_model_${TIMESTAMP}.pt"

echo "================================================================"
echo "AlphaQuant GPTQ Quantization"
echo "================================================================"
echo "Model:        $MODEL"
echo "Config:       $CONFIG"
echo "Dataset:      $DATASET"
echo "Samples:      $NSAMPLES"
echo "Method:       $METHOD"
echo "Output:       $OUTPUT_FILE"
echo "================================================================"
echo ""

# Run GPTQ quantization
python scripts/run_gptq.py \
    --model "$MODEL" \
    --config "$CONFIG" \
    --dataset "$DATASET" \
    --nsamples "$NSAMPLES" \
    --seqlen "$SEQLEN" \
    --method "$METHOD" \
    --blocksize "$BLOCKSIZE" \
    --percdamp "$PERCDAMP" \
    --save "$OUTPUT_FILE"

echo ""
echo "================================================================"
echo "Quantization complete!"
echo "Model saved to: $OUTPUT_FILE"
echo "================================================================"

