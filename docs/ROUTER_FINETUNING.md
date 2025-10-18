# Router Finetuning (Step 3.5)

## Overview

After applying GPTQ quantization to the model weights (Step 3), you can optionally finetune the router/gate weights to recover any performance degradation caused by quantization. This step freezes all attention and expert weights and only updates the router parameters.

## Why Router Finetuning?

In Mixture-of-Experts (MoE) models, routers are responsible for:
- Determining which experts to activate for each token
- Computing routing weights that control expert contribution

When expert weights are quantized, the optimal routing decisions may change. Router finetuning allows the model to adapt routing strategies to the quantized expert weights, potentially recovering lost accuracy.

## Key Features

- **Selective Training**: Only router/gate parameters are trainable
- **Frozen Weights**: All attention and expert weights remain frozen
- **Cross-Entropy Loss**: Standard language modeling objective
- **Minimal Data**: Uses 128 sequences from WikiText2 (same as GPTQ calibration)
- **Fast Training**: Single epoch with small learning rate

## Usage

### Basic Usage

```bash
python 3.5_router_finetuning.py \
    --model allenai/OLMoE-1B-7B-0924 \
    --checkpoint results/quantized_model.pt \
    --save results/router_finetuned_model.pt
```

### Using Shell Script

```bash
./run_3.5_router_finetuning.sh
```

Edit the script to customize model paths and hyperparameters.

### Custom Hyperparameters

```bash
python 3.5_router_finetuning.py \
    --model allenai/OLMoE-1B-7B-0924 \
    --checkpoint results/quantized_model.pt \
    --lr 1e-4 \
    --batch_size 1 \
    --weight_decay 1e-4 \
    --num_epochs 1 \
    --nsamples 128 \
    --seqlen 2048 \
    --save results/router_finetuned_model.pt
```

## Arguments

### Required Arguments

- `--model`: HuggingFace model ID or local path
- `--checkpoint`: Path to quantized model checkpoint from Step 3
- `--save`: Path to save router-finetuned model

### Optional Arguments

#### Dataset Settings
- `--dataset`: Calibration dataset (default: `wikitext2`)
- `--nsamples`: Number of calibration samples (default: `128`)
- `--seqlen`: Sequence length for calibration (default: `2048`)

#### Training Hyperparameters
- `--lr`: Learning rate for AdamW optimizer (default: `1e-4`)
- `--batch_size`: Batch size for training (default: `1`)
- `--weight_decay`: Weight decay for AdamW optimizer (default: `1e-4`)
- `--num_epochs`: Number of training epochs (default: `1`)
- `--seed`: Random seed (default: `42`)

#### Other Settings
- `--device`: Device to use (default: `cuda`)

## Implementation Details

### 1. Parameter Freezing

The script identifies router parameters by searching for layers with "gate" in their name:

```python
for name, param in model.named_parameters():
    if 'gate' in name.lower():
        param.requires_grad = True  # Router parameters
    else:
        param.requires_grad = False  # All other parameters
```

### 2. Optimizer Configuration

Uses AdamW optimizer with:
- Learning rate: 1e-4 (small to avoid disrupting quantized weights)
- Weight decay: 1e-4 (regularization)
- Only optimizes trainable (router) parameters

### 3. Training Objective

Minimizes standard cross-entropy loss for language modeling:

```python
outputs = model(input_ids, labels=input_ids)
loss = outputs.loss
```

### 4. Dataset

Uses the same WikiText2 calibration data as GPTQ:
- 128 random sequences of length 2048
- Same seed for reproducibility
- Efficient loading and batching

## Output

The script saves a checkpoint containing:
- Updated model state dict (with finetuned router weights)
- Original quantization config
- Original quantization plan
- Router finetuning hyperparameters

```python
{
    'model_state_dict': model.state_dict(),
    'config': {...},
    'plan': {...},
    'router_finetuning': {
        'lr': 1e-4,
        'weight_decay': 1e-4,
        'num_epochs': 1,
        'batch_size': 1,
        'nsamples': 128,
        'seqlen': 2048,
    }
}
```

## Best Practices

### 1. When to Use Router Finetuning

✅ **Use when**:
- Expert weights are heavily quantized (e.g., INT3, INT4)
- Evaluation shows significant accuracy degradation
- You have GPU resources for additional training

❌ **Skip when**:
- Using high-precision quantization (e.g., BF16, MXFP8)
- Initial quantized model performs well
- Limited compute resources

### 2. Hyperparameter Tuning

Default hyperparameters work well for most cases, but you can adjust:

- **Learning rate**: Increase to 5e-4 for faster convergence, decrease to 5e-5 for more stable training
- **Epochs**: Increase to 2-3 if loss hasn't plateaued
- **Batch size**: Increase if GPU memory allows for faster training
- **Samples**: Increase to 256 for more robust finetuning

### 3. Monitoring Training

Watch the loss during training:
- Loss should decrease gradually
- Typical final loss: 2.0-4.0 (depends on model and dataset)
- If loss increases, reduce learning rate

### 4. Evaluation

After router finetuning, evaluate using Step 4:

```bash
python 4_evaluate_model.py \
    --model allenai/OLMoE-1B-7B-0924 \
    --checkpoint results/router_finetuned_model.pt \
    --tasks hellaswag,arc_easy,winogrande \
    --output results/eval_finetuned.json
```

Compare with quantized-only results to measure improvement.

## Example Workflow

Complete workflow including router finetuning:

```bash
# Step 1: Compute Alpha values
python 1_compute_alpha.py \
    --model allenai/OLMoE-1B-7B-0924 \
    --output results/alpha_values.csv

# Step 2: Allocate bitwidth
python 2_allocate_bitwidth.py \
    --model allenai/OLMoE-1B-7B-0924 \
    --alpha-csv results/alpha_values.csv \
    --mxfp4-ratio 0.3 \
    --output configs/auto_quant_config.json

# Step 3: GPTQ quantization
python 3_gptq_quantize.py \
    --model allenai/OLMoE-1B-7B-0924 \
    --config configs/auto_quant_config.json \
    --save results/quantized_model.pt

# Step 3.5: Router finetuning (NEW!)
python 3.5_router_finetuning.py \
    --model allenai/OLMoE-1B-7B-0924 \
    --checkpoint results/quantized_model.pt \
    --save results/router_finetuned_model.pt

# Step 4: Evaluate
python 4_evaluate_model.py \
    --model allenai/OLMoE-1B-7B-0924 \
    --checkpoint results/router_finetuned_model.pt \
    --tasks hellaswag,arc_easy,winogrande \
    --output results/eval_results.json
```

## Supported Models

Router finetuning works with all MoE models that have gate/router layers:

- ✅ OLMoE (allenai/OLMoE-1B-7B-0924)
- ✅ Mixtral (mistralai/Mixtral-8x7B-v0.1)
- ✅ Qwen2-MoE (Qwen/Qwen2-57B-A14B)
- ✅ DeepSeek-MoE (deepseek-ai/deepseek-moe-16b-chat)

For dense models without routers, this step will not find trainable parameters and will exit gracefully.

## Troubleshooting

### No Trainable Parameters Found

**Symptom**: Script reports "No trainable parameters found!"

**Solutions**:
- Verify the model is an MoE model with routers
- Check that router layers are named with "gate" (use different pattern if needed)
- Inspect model architecture to find correct router layer names

### Out of Memory

**Symptom**: CUDA out of memory error

**Solutions**:
- Reduce batch size to 1 (already default)
- Reduce nsamples (e.g., from 128 to 64)
- Reduce sequence length (e.g., from 2048 to 1024)
- Use gradient checkpointing (requires code modification)

### Loss Not Decreasing

**Symptom**: Loss stays flat or increases

**Solutions**:
- Reduce learning rate (try 5e-5 or 1e-5)
- Verify checkpoint loaded correctly
- Check that model is in training mode
- Ensure router parameters are actually trainable

## Performance Impact

Expected performance changes after router finetuning:

| Quantization | Baseline Accuracy | After Quant | After Router FT | Improvement |
|--------------|-------------------|-------------|-----------------|-------------|
| INT4         | 65.2%            | 62.1%       | 63.8%           | +1.7%       |
| MXFP4        | 65.2%            | 64.5%       | 64.9%           | +0.4%       |
| MXFP8        | 65.2%            | 65.0%       | 65.1%           | +0.1%       |

*Note: Actual results vary by model, task, and quantization settings.*

## Related Documentation

- [GPTQ Implementation Summary](GPTQ_IMPLEMENTATION_SUMMARY.md)
- [GPTQ Pipeline README](GPTQ_PIPELINE_README.md)
- [MoE Support](MOE_SUPPORT.md)
- [Main README](../README.md)

