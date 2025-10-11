import torch
import torch.nn.functional as F


def fake_quant_int2(x: torch.Tensor, 
                   stochastic_rounding: bool = False,
                   symmetric: bool = True) -> torch.Tensor:
    """
    Fake quantize to INT2 (2-bit integer).
    
    Args:
        x: Input tensor
        stochastic_rounding: Whether to use stochastic rounding
        symmetric: Whether to use symmetric quantization (range: [-2, 1] vs [0, 3])
    """
    if symmetric:
        qmin, qmax = -2, 1
    else:
        qmin, qmax = 0, 3
    
    # Calculate scale and zero point
    x_min = x.min()
    x_max = x.max()
    
    if symmetric:
        scale = torch.max(x_max.abs(), x_min.abs()) / qmax
        zero_point = 0
    else:
        scale = (x_max - x_min) / (qmax - qmin)
        zero_point = qmin - x_min / scale
    
    # Quantize
    x_scaled = x / scale + zero_point
    
    if stochastic_rounding:
        noise = torch.rand_like(x_scaled) - 0.5
        x_scaled = x_scaled + noise
    
    x_quant = torch.clamp(torch.round(x_scaled), qmin, qmax)
    
    # Dequantize
    x_dequant = (x_quant - zero_point) * scale
    
    return x_dequant


def fake_quant_int3(x: torch.Tensor, 
                   stochastic_rounding: bool = False,
                   symmetric: bool = True) -> torch.Tensor:
    """
    Fake quantize to INT3 (3-bit integer).
    
    Args:
        x: Input tensor
        stochastic_rounding: Whether to use stochastic rounding
        symmetric: Whether to use symmetric quantization (range: [-4, 3] vs [0, 7])
    """
    if symmetric:
        qmin, qmax = -4, 3
    else:
        qmin, qmax = 0, 7
    
    # Calculate scale and zero point
    x_min = x.min()
    x_max = x.max()
    
    if symmetric:
        scale = torch.max(x_max.abs(), x_min.abs()) / qmax
        zero_point = 0
    else:
        scale = (x_max - x_min) / (qmax - qmin)
        zero_point = qmin - x_min / scale
    
    # Quantize
    x_scaled = x / scale + zero_point
    
    if stochastic_rounding:
        noise = torch.rand_like(x_scaled) - 0.5
        x_scaled = x_scaled + noise
    
    x_quant = torch.clamp(torch.round(x_scaled), qmin, qmax)
    
    # Dequantize
    x_dequant = (x_quant - zero_point) * scale
    
    return x_dequant


def fake_quant_int4(x: torch.Tensor, 
                   stochastic_rounding: bool = False,
                   symmetric: bool = True) -> torch.Tensor:
    """
    Fake quantize to INT4 (4-bit integer).
    
    Args:
        x: Input tensor
        stochastic_rounding: Whether to use stochastic rounding
        symmetric: Whether to use symmetric quantization (range: [-8, 7] vs [0, 15])
    """
    if symmetric:
        qmin, qmax = -8, 7
    else:
        qmin, qmax = 0, 15
    
    # Calculate scale and zero point
    x_min = x.min()
    x_max = x.max()
    
    if symmetric:
        scale = torch.max(x_max.abs(), x_min.abs()) / qmax
        zero_point = 0
    else:
        scale = (x_max - x_min) / (qmax - qmin)
        zero_point = qmin - x_min / scale
    
    # Quantize
    x_scaled = x / scale + zero_point
    
    if stochastic_rounding:
        noise = torch.rand_like(x_scaled) - 0.5
        x_scaled = x_scaled + noise
    
    x_quant = torch.clamp(torch.round(x_scaled), qmin, qmax)
    
    # Dequantize
    x_dequant = (x_quant - zero_point) * scale
    
    return x_dequant


def fake_quant_int6(x: torch.Tensor, 
                   stochastic_rounding: bool = False,
                   symmetric: bool = True) -> torch.Tensor:
    """
    Fake quantize to INT6 (6-bit integer).
    
    Args:
        x: Input tensor
        stochastic_rounding: Whether to use stochastic rounding
        symmetric: Whether to use symmetric quantization (range: [-32, 31] vs [0, 63])
    """
    if symmetric:
        qmin, qmax = -32, 31
    else:
        qmin, qmax = 0, 63
    
    # Calculate scale and zero point
    x_min = x.min()
    x_max = x.max()
    
    if symmetric:
        scale = torch.max(x_max.abs(), x_min.abs()) / qmax
        zero_point = 0
    else:
        scale = (x_max - x_min) / (qmax - qmin)
        zero_point = qmin - x_min / scale
    
    # Quantize
    x_scaled = x / scale + zero_point
    
    if stochastic_rounding:
        noise = torch.rand_like(x_scaled) - 0.5
        x_scaled = x_scaled + noise
    
    x_quant = torch.clamp(torch.round(x_scaled), qmin, qmax)
    
    # Dequantize
    x_dequant = (x_quant - zero_point) * scale
    
    return x_dequant


def fake_quant_int8(x: torch.Tensor, 
                   stochastic_rounding: bool = False,
                   symmetric: bool = True) -> torch.Tensor:
    """
    Fake quantize to INT8 (8-bit integer).
    
    Args:
        x: Input tensor
        stochastic_rounding: Whether to use stochastic rounding
        symmetric: Whether to use symmetric quantization (range: [-128, 127] vs [0, 255])
    """
    if symmetric:
        qmin, qmax = -128, 127
    else:
        qmin, qmax = 0, 255
    
    # Calculate scale and zero point
    x_min = x.min()
    x_max = x.max()
    
    if symmetric:
        scale = torch.max(x_max.abs(), x_min.abs()) / qmax
        zero_point = 0
    else:
        scale = (x_max - x_min) / (qmax - qmin)
        zero_point = qmin - x_min / scale
    
    # Quantize
    x_scaled = x / scale + zero_point
    
    if stochastic_rounding:
        noise = torch.rand_like(x_scaled) - 0.5
        x_scaled = x_scaled + noise
    
    x_quant = torch.clamp(torch.round(x_scaled), qmin, qmax)
    
    # Dequantize
    x_dequant = (x_quant - zero_point) * scale
    
    return x_dequant


def fake_quant_fp4(x: torch.Tensor, 
                  stochastic_rounding: bool = False,
                  format: str = 'e2m1') -> torch.Tensor:
    """
    Fake quantize to FP4 (4-bit floating point).
    
    Args:
        x: Input tensor
        stochastic_rounding: Whether to use stochastic rounding
        format: FP4 format ('e2m1' for 2 exponent bits, 1 mantissa bit)
    """
    if format == 'e2m1':
        # FP4 E2M1: 1 sign + 2 exponent + 1 mantissa
        # Representable values (positive): 0, 0.5, 1, 1.5, 2, 3, 4, 6
        fp4_values = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], 
                                  device=x.device, dtype=x.dtype)
    else:
        raise ValueError(f"Unsupported FP4 format: {format}")
    
    # Calculate scale
    x_abs = x.abs()
    x_max = x_abs.max()
    scale = fp4_values[-1] / x_max.clamp(min=1e-8)
    
    # Scale to FP4 range
    x_scaled = x * scale
    x_sign = torch.sign(x_scaled)
    x_abs_scaled = x_scaled.abs()
    
    # Quantize to nearest representable FP4 value
    # Find nearest value in fp4_values for each element
    diff = torch.abs(x_abs_scaled.unsqueeze(-1) - fp4_values.unsqueeze(0))
    nearest_idx = torch.argmin(diff, dim=-1)
    x_quant_abs = fp4_values[nearest_idx]
    
    if stochastic_rounding:
        # For stochastic rounding, probabilistically round up/down
        next_idx = torch.clamp(nearest_idx + 1, max=len(fp4_values) - 1)
        prev_idx = torch.clamp(nearest_idx - 1, min=0)
        
        # Randomly choose between nearest and next value
        rand_mask = torch.rand_like(x_abs_scaled) > 0.5
        x_quant_abs = torch.where(rand_mask, fp4_values[next_idx], x_quant_abs)
    
    x_quant = x_sign * x_quant_abs
    
    # Dequantize
    x_dequant = x_quant / scale
    
    return x_dequant


def _generate_fp6_values(format: str = 'e3m2') -> torch.Tensor:
    """Generate all representable FP6 values for a given format."""
    values = [0.0]
    
    if format == 'e2m3':
        # E2M3: 2 exponent bits (bias=1), 3 mantissa bits
        exp_bits = 2
        mant_bits = 3
        bias = 1
        exp_range = 2 ** exp_bits  # 0-3
        mant_range = 2 ** mant_bits  # 0-7
        
        for exp in range(exp_range):
            for mant in range(mant_range):
                if exp == 0:
                    # Subnormal numbers
                    if mant == 0:
                        continue  # Skip duplicate zero
                    value = (2.0 ** (1 - bias)) * (mant / float(mant_range))
                else:
                    # Normal numbers
                    value = (2.0 ** (exp - bias)) * (1.0 + mant / float(mant_range))
                values.append(value)
                
    elif format == 'e3m2':
        # E3M2: 3 exponent bits (bias=3), 2 mantissa bits
        exp_bits = 3
        mant_bits = 2
        bias = 3
        exp_range = 2 ** exp_bits  # 0-7
        mant_range = 2 ** mant_bits  # 0-3
        
        for exp in range(exp_range):
            for mant in range(mant_range):
                if exp == 0:
                    # Subnormal numbers
                    if mant == 0:
                        continue  # Skip duplicate zero
                    value = (2.0 ** (1 - bias)) * (mant / float(mant_range))
                else:
                    # Normal numbers
                    value = (2.0 ** (exp - bias)) * (1.0 + mant / float(mant_range))
                values.append(value)
    else:
        raise ValueError(f"Unsupported FP6 format: {format}")
    
    return torch.tensor(sorted(values), dtype=torch.float32)


def _generate_fp8_values(format: str = 'e4m3') -> torch.Tensor:
    """Generate all representable FP8 values for a given format."""
    if format == 'e4m3':
        # E4M3: 4 exponent bits (bias=7), 3 mantissa bits
        # Values: [0, 0.001953125, 0.00390625, ..., 448]
        values = []
        # Special cases
        values.append(0.0)
        
        # Normalized numbers
        for exp in range(0, 15):  # 4-bit exponent (excluding special values)
            for mant in range(0, 8):  # 3-bit mantissa
                # value = (-1)^sign * 2^(exp - bias) * (1 + mantissa/2^3)
                if exp == 0 and mant == 0:
                    continue  # Skip duplicate zero
                bias = 7
                value = (2.0 ** (exp - bias)) * (1.0 + mant / 8.0)
                if value <= 448.0:  # FP8 E4M3 max value
                    values.append(value)
        
        return torch.tensor(sorted(values), dtype=torch.float32)
    
    elif format == 'e5m2':
        # E5M2: 5 exponent bits (bias=15), 2 mantissa bits
        values = []
        values.append(0.0)
        
        for exp in range(0, 31):
            for mant in range(0, 4):  # 2-bit mantissa
                if exp == 0 and mant == 0:
                    continue
                bias = 15
                value = (2.0 ** (exp - bias)) * (1.0 + mant / 4.0)
                if value <= 57344.0:  # FP8 E5M2 max value
                    values.append(value)
        
        return torch.tensor(sorted(values), dtype=torch.float32)
    
    else:
        raise ValueError(f"Unsupported FP8 format: {format}")


def fake_quant_fp6(x: torch.Tensor, 
                  stochastic_rounding: bool = False,
                  format: str = 'e3m2') -> torch.Tensor:
    """
    Fake quantize to FP6 (6-bit floating point).
    
    Args:
        x: Input tensor
        stochastic_rounding: Whether to use stochastic rounding
        format: FP6 format ('e2m3' or 'e3m2')
    """
    # Generate representable FP6 values
    fp6_values = _generate_fp6_values(format).to(x.device)
    
    # Calculate scale
    x_abs = x.abs()
    x_max = x_abs.max()
    scale = fp6_values[-1] / x_max.clamp(min=1e-8)
    
    # Scale to FP6 range
    x_scaled = x * scale
    x_sign = torch.sign(x_scaled)
    x_abs_scaled = x_scaled.abs()
    
    # Quantize to nearest representable FP6 value
    # For efficiency with large tensors, use searchsorted instead of full distance matrix
    x_abs_flat = x_abs_scaled.flatten()
    
    # Find nearest value using binary search
    indices = torch.searchsorted(fp6_values, x_abs_flat)
    indices = torch.clamp(indices, 0, len(fp6_values) - 1)
    
    # Check both current and previous index to find nearest
    indices_prev = torch.clamp(indices - 1, min=0)
    
    dist_curr = torch.abs(x_abs_flat - fp6_values[indices])
    dist_prev = torch.abs(x_abs_flat - fp6_values[indices_prev])
    
    use_prev = dist_prev < dist_curr
    nearest_idx = torch.where(use_prev, indices_prev, indices)
    
    if stochastic_rounding:
        # Stochastic rounding: probabilistically choose between nearest values
        indices_next = torch.clamp(nearest_idx + 1, max=len(fp6_values) - 1)
        rand_mask = torch.rand_like(x_abs_flat) > 0.5
        nearest_idx = torch.where(rand_mask, indices_next, nearest_idx)
    
    x_quant_abs = fp6_values[nearest_idx].reshape(x.shape)
    x_quant = x_sign * x_quant_abs
    
    # Dequantize
    x_dequant = x_quant / scale
    
    return x_dequant


def fake_quant_fp8(x: torch.Tensor, 
                  stochastic_rounding: bool = False,
                  format: str = 'e4m3') -> torch.Tensor:
    """
    Fake quantize to FP8 (8-bit floating point).
    
    Args:
        x: Input tensor
        stochastic_rounding: Whether to use stochastic rounding
        format: FP8 format ('e4m3' or 'e5m2')
    """
    # Generate representable FP8 values
    fp8_values = _generate_fp8_values(format).to(x.device)
    
    # Calculate scale
    x_abs = x.abs()
    x_max = x_abs.max()
    scale = fp8_values[-1] / x_max.clamp(min=1e-8)
    
    # Scale to FP8 range
    x_scaled = x * scale
    x_sign = torch.sign(x_scaled)
    x_abs_scaled = x_scaled.abs()
    
    # Quantize to nearest representable FP8 value
    # For efficiency with large tensors, use searchsorted instead of full distance matrix
    x_abs_flat = x_abs_scaled.flatten()
    
    # Find nearest value using binary search
    indices = torch.searchsorted(fp8_values, x_abs_flat)
    indices = torch.clamp(indices, 0, len(fp8_values) - 1)
    
    # Check both current and previous index to find nearest
    indices_prev = torch.clamp(indices - 1, min=0)
    
    dist_curr = torch.abs(x_abs_flat - fp8_values[indices])
    dist_prev = torch.abs(x_abs_flat - fp8_values[indices_prev])
    
    use_prev = dist_prev < dist_curr
    nearest_idx = torch.where(use_prev, indices_prev, indices)
    
    if stochastic_rounding:
        # Stochastic rounding: probabilistically choose between nearest values
        indices_next = torch.clamp(nearest_idx + 1, max=len(fp8_values) - 1)
        rand_mask = torch.rand_like(x_abs_flat) > 0.5
        nearest_idx = torch.where(rand_mask, indices_next, nearest_idx)
    
    x_quant_abs = fp8_values[nearest_idx].reshape(x.shape)
    x_quant = x_sign * x_quant_abs
    
    # Dequantize
    x_dequant = x_quant / scale
    
    return x_dequant


if __name__ == '__main__':
    # Test the quantization functions
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test data
    x = torch.randn(2, 32, device=device) * 10
    
    print("Original tensor:", x[0, :8])
    
    # Test INT quantizers
    for bits in [2, 3, 4, 6, 8]:
        quant_func = globals()[f'fake_quant_int{bits}']
        x_quant = quant_func(x, stochastic_rounding=False)
        print(f"INT{bits} quantized:", x_quant[0, :8])
    
    # Test FP quantizers
    # FP4
    for format in ['e2m1']:
        x_quant = fake_quant_fp4(x, stochastic_rounding=False, format=format)
        print(f"FP4-{format} quantized:", x_quant[0, :8])
    
    # FP6
    for format in ['e2m3', 'e3m2']:
        x_quant = fake_quant_fp6(x, stochastic_rounding=False, format=format)
        print(f"FP6-{format} quantized:", x_quant[0, :8])
    
    # FP8
    for format in ['e4m3', 'e5m2']:
        x_quant = fake_quant_fp8(x, stochastic_rounding=False, format=format)
        print(f"FP8-{format} quantized:", x_quant[0, :8])
