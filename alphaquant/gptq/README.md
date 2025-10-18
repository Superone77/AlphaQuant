# GPTQ Quantization for AlphaQuant

This module provides mixed-precision GPTQ quantization capabilities for AlphaQuant, inspired by MoEQuant but fully integrated with AlphaQuant's quantizer system.

## Overview

GPTQ (Generative Pre-trained Transformer Quantization) is a post-training quantization method that uses second-order information (Hessian) to minimize quantization error. This implementation supports:

- **Mixed-precision quantization**: Different layers can use different quantization formats
- **Multiple quantizers**: INT2/3/4/6/8, FP4/6/8, MXFP4/6/8
- **Group-wise quantization**: Configurable group sizes for better accuracy
- **Activation ordering**: Optional feature for improved quantization
- **MoE support**: Special handling for Mixture-of-Experts models

## Architecture

```
alphaquant/gptq/
├── __init__.py           # Module exports
├── gptq.py              # Core GPTQ algorithm
├── quantize.py          # Main quantization pipeline
├── data_utils.py        # Calibration data loaders
├── model_utils.py       # Model structure utilities
└── README.md            # This file
```

## Quick Start

### 1. Basic Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from alphaquant.gptq import gptq_quantize_model, GPTQConfig
from alphaquant.gptq.data_utils import CalibrationDataLoader
from alphaquant.utils.replacement import load_layer_config, plan_model_layer_schemes

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.bfloat16,
    device_map='cpu'
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# Load quantization config
layer_config_raw = load_layer_config('configs/gptq_example.json')
plans = plan_model_layer_schemes(model, layer_config_raw)
layer_config = {name: scheme for name, scheme in plans}

# Prepare calibration data
dataloader = CalibrationDataLoader(
    dataset_name='wikitext2',
    nsamples=128,
    seqlen=2048,
    tokenizer=tokenizer
)

# Configure GPTQ
gptq_config = GPTQConfig(
    blocksize=128,
    percdamp=0.01,
    actorder=False
)

# Quantize
quantizers = gptq_quantize_model(
    model=model,
    dataloader=dataloader,
    layer_config=layer_config,
    gptq_config=gptq_config,
    device='cuda'
)
```

### 2. Using the CLI Script

```bash
python scripts/run_gptq.py \
    --model meta-llama/Llama-2-7b-hf \
    --config configs/gptq_mixed_precision.json \
    --dataset wikitext2 \
    --nsamples 128 \
    --method gptq \
    --save quantized_model.pt
```

## Configuration Format

Quantization configurations are specified in JSON format:

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
      "aq": "bf16",
      "group_size": 128
    },
    {
      "pattern": "model.layers.*.mlp.down_proj",
      "wq": "int3",
      "aq": "bf16",
      "group_size": 128
    }
  ]
}
```

### Configuration Fields

- `default`: Default quantization scheme for all layers
  - `wq`: Weight quantization format
  - `aq`: Activation quantization format
  - `group_size`: Group size for group-wise quantization
  - `extra`: Additional format-specific parameters

- `overrides`: Layer-specific overrides
  - `pattern`: Glob pattern to match layer names
  - `wq`, `aq`, `group_size`: Override values
  - `skip`: Set to `true` to skip quantizing this layer

### Supported Quantization Formats

- **Integer**: `int2`, `int3`, `int4`, `int6`, `int8`
- **Floating Point**: `fp4`, `fp6`, `fp8`
- **Microscaling Float**: `mxfp4`, `mxfp6`, `mxfp8`
- **No Quantization**: `bf16` (bfloat16)

## Example Configurations

### 1. Uniform INT4 Quantization

See `configs/gptq_example.json` - all layers quantized to INT4 with group size 128.

### 2. Mixed-Precision Quantization

See `configs/gptq_mixed_precision.json`:
- First layer: INT6 (higher precision)
- Attention layers: INT4
- MLP gate/up: MXFP4
- MLP down: INT3 (lower precision)

### 3. MoE-Specific Configuration

See `configs/gptq_olmoe_mixed.json`:
- Attention Q/K/V: MXFP6
- Attention O: INT4
- Expert FFNs: MXFP4
- Shared expert: MXFP6
- Router: Skip quantization

## GPTQ Algorithm Parameters

### GPTQConfig Options

```python
GPTQConfig(
    blocksize=128,      # Block size for column processing
    percdamp=0.01,      # Dampening percentage for Hessian
    groupsize=-1,       # Group size (-1 = per-channel, >0 = group-wise)
    actorder=False,     # Use activation ordering
    static_groups=False # Use static group boundaries
)
```

- **blocksize**: Larger values are faster but use more memory
- **percdamp**: Higher values improve numerical stability
- **groupsize**: Smaller groups can improve accuracy but increase overhead
- **actorder**: Can improve accuracy but is slower
- **static_groups**: Pre-compute group boundaries (faster but less flexible)

## Calibration Data

### Supported Datasets

- **WikiText-2**: Standard language modeling benchmark
- **C4**: Larger and more diverse text corpus

### Custom Data

You can provide custom calibration data:

```python
class CustomDataLoader:
    def __iter__(self):
        for sample in my_data:
            yield sample  # torch.Tensor of shape [1, seqlen]
```

## API Reference

### Main Functions

#### `gptq_quantize_model`

```python
def gptq_quantize_model(
    model: nn.Module,
    dataloader: Iterator[torch.Tensor],
    layer_config: Dict[str, Dict[str, Any]],
    device: str = 'cuda',
    gptq_config: Optional[GPTQConfig] = None,
    model_type: str = 'auto',
    dtype: str = 'bfloat16'
) -> Dict[str, Any]
```

Quantize a model using GPTQ.

**Returns**: Dictionary of quantizers for each layer

#### `rtn_quantize_model`

```python
def rtn_quantize_model(
    model: nn.Module,
    layer_config: Dict[str, Dict[str, Any]],
    device: str = 'cuda',
    model_type: str = 'auto',
    dtype: str = 'bfloat16'
) -> Dict[str, Any]
```

Quantize a model using RTN (Round-to-Nearest) - no calibration needed.

**Returns**: Dictionary of quantizers for each layer

## Performance Tips

1. **Memory Management**
   - Process one layer at a time to reduce memory usage
   - Use CPU offloading for very large models
   - Reduce `nsamples` if running out of memory

2. **Speed Optimization**
   - Increase `blocksize` for faster processing
   - Disable `actorder` for speed
   - Use RTN instead of GPTQ for quick experiments

3. **Accuracy Optimization**
   - Increase `nsamples` for better calibration
   - Use smaller `group_size` for sensitive layers
   - Enable `actorder` for critical layers
   - Use higher precision for first/last layers

## Comparison with MoEQuant

This implementation differs from MoEQuant in several ways:

| Feature | MoEQuant | AlphaQuant GPTQ |
|---------|----------|-----------------|
| Quantizer System | Custom | AlphaQuant's unified system |
| Configuration | Code-based | JSON-based |
| Format Support | INT only | INT, FP, MXFP |
| MoE Handling | Built-in | Via configuration |
| Integration | Standalone | Part of AlphaQuant |

## Troubleshooting

### Common Issues

**Issue**: Out of memory during quantization
- **Solution**: Reduce `nsamples`, increase `blocksize`, or use CPU offloading

**Issue**: NaN values in quantized weights
- **Solution**: Increase `percdamp`, check input data, or skip problematic layers

**Issue**: Poor accuracy after quantization
- **Solution**: Increase `nsamples`, use higher precision for sensitive layers, or reduce `group_size`

**Issue**: Layer pattern not matching
- **Solution**: Check layer names with `model.named_modules()`, adjust glob patterns

## Examples

### Example 1: Quantize Llama-2-7B

```bash
python scripts/run_gptq.py \
    --model meta-llama/Llama-2-7b-hf \
    --config configs/gptq_mixed_precision.json \
    --dataset wikitext2 \
    --nsamples 128 \
    --blocksize 128 \
    --save llama2-7b-gptq.pt
```

### Example 2: Quantize OLMoE with Mixed Precision

```bash
python scripts/run_gptq.py \
    --model allenai/OLMoE-1B-7B-0924 \
    --config configs/gptq_olmoe_mixed.json \
    --dataset c4 \
    --nsamples 256 \
    --actorder \
    --save olmoe-gptq.pt
```

### Example 3: RTN (No Calibration)

```bash
python scripts/run_gptq.py \
    --model meta-llama/Llama-2-7b-hf \
    --config configs/gptq_example.json \
    --method rtn \
    --save llama2-7b-rtn.pt
```

## Citation

If you use this GPTQ implementation, please consider citing:

```bibtex
@article{frantar2022gptq,
  title={GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers},
  author={Frantar, Elias and Ashkboos, Saleh and Hoefler, Torsten and Alistarh, Dan},
  journal={arXiv preprint arXiv:2210.17323},
  year={2022}
}
```

## License

This module is part of AlphaQuant and follows the same license terms.

