#!/bin/bash
#
# Run Step 3.5: Router Finetuning
#
# This script finetunes only the router weights of the quantized model.
# All attention and expert weights are frozen.
#
# Usage:
#   ./run_3.5_router_finetuning.sh

set -e  # Exit on error

echo "========================================"
echo "Step 3.5: Router Finetuning"
echo "========================================"

# Configuration
MODEL="allenai/OLMoE-1B-7B-0924"
CHECKPOINT="outputs/quantized_model.pt"
SAVE_PATH="outputs/router_finetuned_model.pt"
DATASET="wikitext2"
NSAMPLES=128
SEQLEN=2048

# Training hyperparameters
LR=1e-4
BATCH_SIZE=1
WEIGHT_DECAY=1e-4
NUM_EPOCHS=1
SEED=42

# Device
DEVICE="cuda"

# Run router finetuning
python 3.5_router_finetuning.py \
    --model ${MODEL} \
    --checkpoint ${CHECKPOINT} \
    --dataset ${DATASET} \
    --nsamples ${NSAMPLES} \
    --seqlen ${SEQLEN} \
    --lr ${LR} \
    --batch_size ${BATCH_SIZE} \
    --weight_decay ${WEIGHT_DECAY} \
    --num_epochs ${NUM_EPOCHS} \
    --seed ${SEED} \
    --save ${SAVE_PATH} \
    --device ${DEVICE}

echo ""
echo "✓ Router finetuning complete!"
echo "Output saved to: ${SAVE_PATH}"
echo ""
echo "Next step: Run 4_evaluate_model.py to evaluate the finetuned model"

