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
        # Range: approximately [-6, 6]
        fp4_max = 6.0
    else:
        raise ValueError(f"Unsupported FP4 format: {format}")
    
    # Calculate scale
    x_abs = x.abs()
    scale = fp4_max / x_abs.max()
    scale = torch.clamp(scale, min=1e-8)
    
    # Scale to FP4 range
    x_scaled = x * scale
    
    if stochastic_rounding:
        noise = torch.rand_like(x_scaled) - 0.5
        x_scaled = x_scaled + noise
    
    # Simple FP4 quantization (simplified)
    # In practice, this would involve proper FP4 bit manipulation
    x_quant = torch.clamp(x_scaled, -fp4_max, fp4_max)
    
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
    if format == 'e4m3':
        # FP8 E4M3: 1 sign + 4 exponent + 3 mantissa
        fp8_max = 448.0
    elif format == 'e5m2':
        # FP8 E5M2: 1 sign + 5 exponent + 2 mantissa
        fp8_max = 57344.0
    else:
        raise ValueError(f"Unsupported FP8 format: {format}")
    
    # Calculate scale
    x_abs = x.abs()
    scale = fp8_max / x_abs.max()
    scale = torch.clamp(scale, min=1e-8)
    
    # Scale to FP8 range
    x_scaled = x * scale
    
    if stochastic_rounding:
        noise = torch.rand_like(x_scaled) - 0.5
        x_scaled = x_scaled + noise
    
    # Simple FP8 quantization (simplified)
    # In practice, this would involve proper FP8 bit manipulation
    x_quant = torch.clamp(x_scaled, -fp8_max, fp8_max)
    
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
    for bits in [2, 4, 6, 8]:
        quant_func = globals()[f'fake_quant_int{bits}']
        x_quant = quant_func(x, stochastic_rounding=False)
        print(f"INT{bits} quantized:", x_quant[0, :8])
    
    # Test FP quantizers
    for format in ['e2m1', 'e4m3', 'e5m2']:
        if format == 'e2m1':
            quant_func = fake_quant_fp4
        else:
            quant_func = fake_quant_fp8
        x_quant = quant_func(x, stochastic_rounding=False, format=format)
        print(f"FP{format} quantized:", x_quant[0, :8])
