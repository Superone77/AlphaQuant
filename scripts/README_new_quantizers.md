# New Quantizers Implementation

This document describes the newly implemented quantizers for FP8, FP4, INT2, INT4, INT6, and INT8 quantization.

## Overview

The following quantizers have been implemented:

### Floating Point Quantizers
- **FP8**: Standard 8-bit floating point quantization (E4M3 and E5M2 formats)
- **FP4**: Standard 4-bit floating point quantization (E2M1 format)

### Integer Quantizers  
- **INT8**: 8-bit integer quantization (symmetric: [-128, 127], asymmetric: [0, 255])
- **INT6**: 6-bit integer quantization (symmetric: [-32, 31], asymmetric: [0, 63])
- **INT4**: 4-bit integer quantization (symmetric: [-8, 7], asymmetric: [0, 15])
- **INT2**: 2-bit integer quantization (symmetric: [-2, 1], asymmetric: [0, 3])

## File Structure

```
alphaquant/quantizers/
├── fp4.py                    # FP4 quantizer implementation
├── fp8.py                    # FP8 quantizer implementation
├── int_quantizers.py         # Unified integer quantizers (INT2, INT4, INT6, INT8)
├── kernel/
│   └── int_torch.py          # Integer quantization kernels
└── __init__.py               # Updated exports

scripts/
├── test_new_quantizers.py    # Comprehensive test suite
├── example_new_quantizers.py # Usage examples
└── README_new_quantizers.md  # This documentation

configs/
└── example_new_quantizers.json # Example quantization plan
```

## Usage

### Basic Usage

```python
from alphaquant.quantizers import INT8Quantizer, INT8Config, FP8Quantizer, FP8Config

# Create INT8 quantizer
int8_config = INT8Config(symmetric=True)
int8_quantizer = INT8Quantizer(int8_config)

# Quantize weights
quantized_weights = int8_quantizer.quantize_weight(weights)

# Quantize activations
quantized_activations = int8_quantizer.quantize_activation(activations)
```

### Using with Observers (Static Quantization)

```python
from alphaquant.quantizers import INT8Quantizer, INT8Config, MinMaxObserver

# Create quantizer with observer
config = INT8Config(symmetric=True)
quantizer = INT8Quantizer(config)
observer = MinMaxObserver()
quantizer.attach_observer(observer)

# Calibrate on data
for data in calibration_data:
    observer.observe(data)

# Finish calibration
quantizer.calibrate_finish()

# Now quantizer uses precomputed scales
quantized = quantizer.quantize_activation(new_data)
```

### Using with Quantization Plans

```python
from alphaquant.utils.replacement import create_quantizer_from_scheme

# Create quantizers through registry
(WQ, WCfg), (AQ, ACfg) = create_quantizer_from_scheme(
    {'wq': 'int8', 'aq': 'fp8'}, 'bfloat16'
)

# Create instances
w_quantizer = WQ(WCfg())
a_quantizer = AQ(ACfg())
```

## Configuration Options

### Integer Quantizers (INT2, INT4, INT6, INT8)

```python
config = INT8Config(
    symmetric=True,      # Use symmetric quantization (default: True)
    group_size=32,       # Group size for quantization
    dtype="bfloat16"     # Compute dtype after dequantization
)
```

### Floating Point Quantizers (FP4, FP8)

```python
# FP4 configuration
fp4_config = FP4Config(
    format="e2m1",       # FP4 format (only e2m1 supported)
    group_size=32,
    dtype="bfloat16"
)

# FP8 configuration  
fp8_config = FP8Config(
    format="e4m3",       # FP8 format: "e4m3" or "e5m2"
    group_size=32,
    dtype="bfloat16"
)
```

## Quantization Plans

You can use the new quantizers in JSON quantization plans:

```json
{
  "default": {
    "wq": "int8",
    "aq": "int8", 
    "group_size": 32
  },
  "overrides": [
    {
      "pattern": "model.layers.*.attention.*",
      "wq": "fp8",
      "aq": "fp8",
      "group_size": 64
    },
    {
      "pattern": "model.layers.*.mlp.*",
      "wq": "int4",
      "aq": "int8"
    }
  ]
}
```

## Testing

Run the comprehensive test suite:

```bash
python scripts/test_new_quantizers.py
```

Run the examples:

```bash
python scripts/example_new_quantizers.py
```

## Implementation Details

### Integer Quantization

- **Symmetric**: Uses range [-2^(n-1), 2^(n-1)-1] for n-bit quantization
- **Asymmetric**: Uses range [0, 2^n-1] for n-bit quantization
- **Scaling**: Calculated as `scale = max_val / qmax` for symmetric, `scale = (max - min) / (qmax - qmin)` for asymmetric
- **Zero Point**: 0 for symmetric, calculated as `qmin - min / scale` for asymmetric

### Floating Point Quantization

- **FP8 E4M3**: 1 sign + 4 exponent + 3 mantissa bits, range ~[-448, 448]
- **FP8 E5M2**: 1 sign + 5 exponent + 2 mantissa bits, range ~[-57344, 57344]  
- **FP4 E2M1**: 1 sign + 2 exponent + 1 mantissa bit, range ~[-6, 6]

### Performance Considerations

- All quantizers support both CPU and CUDA tensors
- Integer quantizers use efficient PyTorch operations
- Floating point quantizers use simplified implementations (can be optimized with proper bit manipulation)
- All quantizers support stochastic rounding for training

## Integration with Existing Code

The new quantizers are fully integrated with the existing AlphaQuant framework:

1. **Registry System**: All quantizers are registered in `replacement.py`
2. **Module System**: Compatible with `QuantLinear` modules
3. **Observer System**: Work with existing observers like `MinMaxObserver`
4. **Configuration System**: Support all standard configuration options

## Future Improvements

1. **Triton Kernels**: Add optimized Triton kernels for integer quantization
2. **Bit Manipulation**: Implement proper bit manipulation for FP4/FP8
3. **Per-Channel Quantization**: Add support for per-channel quantization
4. **Mixed Precision**: Enhanced support for mixed precision quantization
5. **Hardware Acceleration**: Optimize for specific hardware (e.g., Tensor Cores)

## Troubleshooting

### Common Issues

1. **CUDA Not Available**: All quantizers work on CPU, but CUDA is recommended for performance
2. **Memory Issues**: Lower bit quantizers (INT2, INT4) may have precision issues with very large models
3. **Scale Calculation**: Ensure input tensors have reasonable ranges to avoid overflow/underflow

### Debug Mode

Enable debug output by setting environment variables:

```bash
export ALPHAQUANT_DEBUG=1
python your_script.py
```

## Contributing

When adding new quantizers:

1. Follow the existing pattern in `base.py`
2. Implement both `quantize_weight` and `quantize_activation` methods
3. Add proper configuration classes
4. Update the registry in `replacement.py`
5. Add comprehensive tests
6. Update this documentation
