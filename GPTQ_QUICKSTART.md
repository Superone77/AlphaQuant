# GPTQ Quantization - Quick Start Guide

This guide will get you started with GPTQ quantization in AlphaQuant in 5 minutes.

## Prerequisites

```bash
# Install required packages
pip install transformers datasets torch
```

## Option 1: Use the Example Script (Easiest)

```bash
python examples/gptq_quantization_example.py
```

This will:
1. Load a model (Llama-2-7B by default)
2. Apply mixed-precision GPTQ quantization
3. Save the quantized model
4. Test generation

**Customize the model**:
Edit the `MODEL_NAME` variable in the script.

## Option 2: Use the CLI (Most Flexible)

### Step 1: Choose a Configuration

We provide three ready-to-use configs:

1. **Basic INT4** (`configs/gptq_example.json`)
   - All layers → INT4
   - Good starting point

2. **Mixed-Precision** (`configs/gptq_mixed_precision.json`)
   - First layer → INT6
   - Attention → INT4
   - MLP gate/up → MXFP4
   - MLP down → INT3

3. **MoE-Optimized** (`configs/gptq_olmoe_mixed.json`)
   - Designed for OLMoE, Mixtral, etc.
   - Experts → MXFP4
   - Shared expert → MXFP6

### Step 2: Run Quantization

```bash
python scripts/run_gptq.py \
    --model meta-llama/Llama-2-7b-hf \
    --config configs/gptq_mixed_precision.json \
    --dataset wikitext2 \
    --nsamples 128 \
    --save quantized_model.pt
```

**Parameters**:
- `--model`: HuggingFace model name or local path
- `--config`: Configuration file
- `--dataset`: Calibration dataset (`wikitext2` or `c4`)
- `--nsamples`: Number of calibration samples
- `--save`: Where to save the quantized model

### Step 3: Load and Use

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# Load quantized weights
checkpoint = torch.load("quantized_model.pt")
model.load_state_dict(checkpoint['model'])

# Use the model
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
inputs = tokenizer("Hello, world!", return_tensors="pt")
outputs = model.generate(**inputs)
```

## Option 3: Use the Shell Script

```bash
# Make it executable (first time only)
chmod +x run_gptq_example.sh

# Run with defaults (Llama-2-7B, mixed-precision)
./run_gptq_example.sh

# Or customize
./run_gptq_example.sh "your-model-name" "path/to/config.json"
```

## Create Your Own Configuration

Create a JSON file:

```json
{
  "default": {
    "wq": "int4",
    "aq": "bf16",
    "group_size": 128
  },
  "overrides": [
    {
      "pattern": "model.layers.0.*",
      "wq": "int6",
      "comment": "First layer needs higher precision"
    },
    {
      "pattern": "model.layers.*.mlp.down_proj",
      "wq": "int3",
      "comment": "Down projection can use lower precision"
    }
  ]
}
```

**Available formats**:
- `int2`, `int3`, `int4`, `int6`, `int8`
- `fp4`, `fp6`, `fp8`
- `mxfp4`, `mxfp6`, `mxfp8`
- `bf16` (no quantization)

## Common Use Cases

### Use Case 1: Maximum Compression

```json
{
  "default": {"wq": "int3", "aq": "bf16", "group_size": 128}
}
```

**Result**: ~3-4x compression, some accuracy loss

### Use Case 2: Balanced Quality/Size

```json
{
  "default": {"wq": "int4", "aq": "bf16", "group_size": 128}
}
```

**Result**: ~4x compression, minimal accuracy loss

### Use Case 3: High Quality

```json
{
  "default": {"wq": "mxfp6", "aq": "bf16", "group_size": 64}
}
```

**Result**: ~2.5x compression, negligible accuracy loss

### Use Case 4: MoE Models

```json
{
  "default": {"wq": "int4", "aq": "bf16", "group_size": 128},
  "overrides": [
    {"pattern": "*.mlp.experts.*", "wq": "mxfp4", "group_size": 32},
    {"pattern": "*.mlp.gate", "skip": true}
  ]
}
```

**Result**: Optimized for Mixture-of-Experts models

## Performance Tips

### Faster Quantization

```bash
python scripts/run_gptq.py \
    --model your-model \
    --config your-config.json \
    --nsamples 64 \           # Fewer samples (default: 128)
    --blocksize 256           # Larger blocks (default: 128)
```

### Better Accuracy

```bash
python scripts/run_gptq.py \
    --model your-model \
    --config your-config.json \
    --nsamples 256 \          # More samples
    --actorder                # Enable activation ordering (slower)
```

### Less Memory

```bash
python scripts/run_gptq.py \
    --model your-model \
    --config your-config.json \
    --nsamples 64 \           # Fewer samples
    --dtype float16           # Use FP16 instead of BF16
```

## Troubleshooting

### Problem: Out of Memory

**Solution 1**: Reduce samples
```bash
--nsamples 64  # or even 32
```

**Solution 2**: Use CPU for model loading
```bash
--device cpu  # then move layers to GPU during quantization
```

### Problem: Poor Accuracy

**Solution 1**: Use more samples
```bash
--nsamples 256  # or 512
```

**Solution 2**: Use higher precision for sensitive layers
```json
{
  "overrides": [
    {"pattern": "model.layers.0.*", "wq": "int6"},
    {"pattern": "model.layers.*.self_attn.*", "wq": "int4"}
  ]
}
```

### Problem: Slow Quantization

**Solution**: Use RTN instead of GPTQ for quick experiments
```bash
--method rtn  # Much faster, slightly lower quality
```

## Next Steps

1. **Read the full documentation**: [GPTQ_PIPELINE_README.md](GPTQ_PIPELINE_README.md)
2. **Explore configurations**: Check `configs/` directory
3. **Try different models**: Test on your own models
4. **Optimize configs**: Experiment with different quantization formats

## Examples

### Llama-2-7B with INT4

```bash
python scripts/run_gptq.py \
    --model meta-llama/Llama-2-7b-hf \
    --config configs/gptq_example.json \
    --save llama2-int4.pt
```

### OLMoE with Mixed Precision

```bash
python scripts/run_gptq.py \
    --model allenai/OLMoE-1B-7B-0924 \
    --config configs/gptq_olmoe_mixed.json \
    --nsamples 256 \
    --save olmoe-mixed.pt
```

### Quick Test with RTN

```bash
python scripts/run_gptq.py \
    --model meta-llama/Llama-2-7b-hf \
    --config configs/gptq_example.json \
    --method rtn \
    --save llama2-rtn.pt
```

---

**Need help?** Check the [full documentation](GPTQ_PIPELINE_README.md) or open an issue.

