import torch
from .nvfp4_triton import nvfp4_forward
from .mxfp4_triton import mxfp4_forward
from .mxfp8_triton import mxfp8_forward


def fp4_121_positive(x:torch.Tensor, stochastic_rounding:bool=False) -> torch.Tensor:
    if stochastic_rounding:
        noise = torch.rand_like(x) - 0.5
        step1 = torch.round(2.0 * x + noise) / 2.0
        step2 = torch.round(x + noise)
        step3 = 2.0 * torch.round(x / 2.0 + noise)
    else:
        step1 = torch.round(2.0 * x) / 2.0
        step2 = torch.round(x)
        step3 = 2.0 * torch.round(x / 2.0)
    
    mask1 = x < 2.0
    mask2 = x < 4.0

    return step1 * mask1 + step2 * (~mask1) * mask2 + step3 * (~mask1) * (~mask2)


def ue5m3(x:torch.Tensor) -> torch.Tensor:
    # NOTE: Assume that array values are in [0, 114688]. (14*2**13 = 114688)
    mask = x <= 2**(-17)
    x_1 = x * mask
    x_2 = x * (~mask) + torch.ones_like(x) * mask

    x_1 = torch.round(x_1 / 2**(-17)) * (2**(-17))

    e = torch.floor(torch.log2(x_2)) - 3
    s = 2**e
    x_2 = torch.round(x_2 / s) * s

    return x_1 * mask + x_2 * (~mask)


FP8_E4M3_MAX = 448.0
def fp4_121_scaled(x:torch.Tensor, 
                   stochastic_rounding:bool=False, 
                   scale_format:str='e8m0') -> torch.Tensor:
    fp4_121_max = 6.0
    sign = x.sign()
    x_abs = x.abs()
    if scale_format == 'e8m0':
        scale = torch.pow(2.0, torch.floor(torch.log2(fp4_121_max / x_abs.max(dim=-1, keepdim=True)[0])))
    
    elif scale_format == 'e4m3':
        nvfp4_max = fp4_121_max * FP8_E4M3_MAX
        scale_per_t = x_abs.max() / nvfp4_max
        x_abs_scaled = x_abs / scale_per_t

        scale_per_b = x_abs_scaled.max(dim=-1, keepdim=True)[0]
        input_tensor = fp4_121_max / scale_per_b
        down_cast = input_tensor.to(torch.float8_e4m3fn)
        # down_cast = torch.ops.hpu.cast_to_fp8_v2(fp4_121_max / scale_per_b, 1.0, False, False, torch.float8_e4m3fn)[0]
        up_cast = down_cast.to(scale_per_b.dtype)
        scale_per_b = up_cast
        scale_per_b = torch.where((0 < scale_per_b) * (scale_per_b < torch.inf), scale_per_b, 1.0)

        x_fp4_abs = fp4_121_positive(x_abs_scaled * scale_per_b, stochastic_rounding) / scale_per_b

        return sign * x_fp4_abs * scale_per_t
    
    elif scale_format == 'ue5m3':
        UE5M3_MAX = 114688.0
        nvfp4_max = fp4_121_max * UE5M3_MAX
        scale_per_t = x_abs.max() / nvfp4_max
        x_abs_scaled = x_abs / scale_per_t

        scale_per_b = x_abs_scaled.max(dim=-1, keepdim=True)[0]

        scale_per_b = ue5m3(fp4_121_max / scale_per_b)
        
        scale_per_b = torch.where((0 < scale_per_b) * (scale_per_b < torch.inf), scale_per_b, 1.0)

        x_fp4_abs = fp4_121_positive(x_abs_scaled * scale_per_b, stochastic_rounding) / scale_per_b

        return sign * x_fp4_abs * scale_per_t

    
    else: # scale_format == 'bf16'
        scale = fp4_121_max / x_abs.max(dim=-1, keepdim=True)[0]

    scale = torch.where((0 < scale) * (scale < torch.inf), scale, 1.0)
    x_fp4_abs = fp4_121_positive(x_abs * scale, stochastic_rounding) / scale
    return sign * x_fp4_abs


def fake_quant_fp4_torch(x:torch.Tensor, 
                   stochastic_rounding:bool=False, 
                   dim:int=-1, 
                   format:str='fp4_e2m1',
                   block_size:int=32, 
                   scale_format:str='e8m0',
                   grid:bool=False) -> torch.Tensor:
    # TODO:
    # 1) enable dim
    # 2) enable e3m0
    shape = x.shape
    if grid:
        assert len(shape) == 2, 'grid enabled for 2d tensors only'
        x = x.reshape(shape[0] // block_size, block_size, shape[1] // block_size, block_size).permute(0, 2, 1, 3).reshape(-1, block_size * block_size)
    else:
        x = x.reshape(-1, block_size)
    
    x = fp4_121_scaled(x, stochastic_rounding, scale_format)
    
    if grid:
        x = x.reshape(shape[0] // block_size, shape[1] // block_size, block_size, block_size).permute(0, 2, 1, 3).reshape(shape)
    else:
        x = x.reshape(shape)
    
    return x
## Triton
def fake_quant_fp4(x:torch.Tensor, 
                   stochastic_rounding:bool=False, 
                   dim:int=-1, 
                   format:str='fp4_e2m1',
                   block_size:int=32, 
                   scale_format:str='e8m0',
                   grid:bool=False) -> torch.Tensor:
    # TODO:
    # 1) enable dim
    # 2) enable e3m0
    x = nvfp4_forward(x, None, stochastic_rounding)
    
    return x


def fake_quant_mxfp4(x:torch.Tensor, 
                     stochastic_rounding:bool=False, 
                     dim:int=-1, 
                     format:str='fp4_e2m1',
                     block_size:int=32, 
                     scale_format:str='e8m0',
                     grid:bool=False) -> torch.Tensor:
    """
    使用MXFP4 triton kernel进行量化
    """
    return mxfp4_forward(x, stochastic_rounding)


def fake_quant_mxfp8(x:torch.Tensor, 
                     stochastic_rounding:bool=False, 
                     format:str='e4m3') -> torch.Tensor:
    """
    使用MXFP8 triton kernel进行量化
    format: 'e4m3' 或 'e5m2'
    """
    return mxfp8_forward(x, format, stochastic_rounding)


def _generate_fp6_values(format: str = 'e3m2') -> torch.Tensor:
    """
    Generate all representable positive FP6 values for a given format.
    FP6 format: 1 sign bit + exponent bits + mantissa bits
    
    E2M3: 2 exponent bits (bias=1), 3 mantissa bits
    E3M2: 3 exponent bits (bias=3), 2 mantissa bits
    """
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


def fp6_quantize_positive(x: torch.Tensor, stochastic_rounding: bool = False, format: str = 'e3m2') -> torch.Tensor:
    """
    Quantize positive values to FP6 format.
    
    Args:
        x: Input tensor (assumed to be non-negative)
        stochastic_rounding: Whether to use stochastic rounding
        format: 'e2m3' or 'e3m2'
    
    Returns:
        Quantized tensor
    """
    if format == 'e2m3':
        # E2M3: exp_bits=2 (bias=1), mant_bits=3
        exp_bits = 2
        mant_bits = 3
        bias = 1
    elif format == 'e3m2':
        # E3M2: exp_bits=3 (bias=3), mant_bits=2
        exp_bits = 3
        mant_bits = 2
        bias = 3
    else:
        raise ValueError(f"Unsupported FP6 format: {format}")
    
    # Get FP6 representable values
    fp6_values = _generate_fp6_values(format).to(x.device).to(x.dtype)
    
    # Handle zeros
    x_nonzero = x.clamp(min=1e-38)
    
    # Compute exponent and mantissa
    log2_x = torch.log2(x_nonzero)
    exponent = torch.floor(log2_x)
    
    # Clamp exponent to valid range
    exp_max = (2 ** exp_bits) - 1
    exp_min = 0
    
    # For subnormals (exp = 0)
    # value = 2^(1-bias) * (mantissa / 2^mant_bits)
    # For normals (exp >= 1)
    # value = 2^(exp-bias) * (1 + mantissa / 2^mant_bits)
    
    # Calculate which exponent bin this value falls into
    exp_unbiased = exponent
    exp_biased = exp_unbiased + bias
    
    # Clamp to valid exponent range
    exp_biased = torch.clamp(exp_biased, exp_min, exp_max)
    
    # Calculate mantissa
    is_subnormal = exp_biased == 0
    
    # For normal numbers
    scale_normal = torch.pow(2.0, exp_biased - bias)
    mantissa_float_normal = (x_nonzero / scale_normal) - 1.0
    mantissa_float_normal = torch.clamp(mantissa_float_normal, 0.0, 1.0 - 1e-6)
    
    # For subnormal numbers (use Python pow for scalar exponent)
    scale_subnormal = (2.0 ** (1 - bias))
    mantissa_float_subnormal = x_nonzero / scale_subnormal
    mantissa_float_subnormal = torch.clamp(mantissa_float_subnormal, 0.0, 1.0 - 1e-6)
    
    # Choose mantissa based on whether value is subnormal
    mantissa_float = torch.where(is_subnormal, mantissa_float_subnormal, mantissa_float_normal)
    
    # Quantize mantissa
    mantissa_levels = 2 ** mant_bits
    mantissa_scaled = mantissa_float * mantissa_levels
    
    if stochastic_rounding:
        noise = torch.rand_like(mantissa_scaled) - 0.5
        mantissa_scaled = mantissa_scaled + noise
    
    mantissa_quant = torch.clamp(torch.round(mantissa_scaled), 0, mantissa_levels - 1)
    
    # Reconstruct value
    mantissa_dequant = mantissa_quant / mantissa_levels
    
    # Reconstruct final value
    value_normal = scale_normal * (1.0 + mantissa_dequant)
    value_subnormal = scale_subnormal * mantissa_dequant
    
    result = torch.where(is_subnormal, value_subnormal, value_normal)
    
    # Handle original zeros
    result = torch.where(x == 0, torch.zeros_like(result), result)
    
    return result


def fp6_scaled(x: torch.Tensor, stochastic_rounding: bool = False, format: str = 'e3m2') -> torch.Tensor:
    """
    FP6 quantization with block-wise scaling (for MXFP6).
    Similar to fp4_121_scaled but for FP6 format.
    
    Args:
        x: Input tensor
        stochastic_rounding: Whether to use stochastic rounding
        format: 'e2m3' or 'e3m2'
    
    Returns:
        Quantized tensor
    """
    # Get FP6 max value based on format
    if format == 'e2m3':
        # E2M3: max = 2^(2) * (1 + 7/8) = 7.5
        fp6_max = 7.5
    elif format == 'e3m2':
        # E3M2: max = 2^(4) * (1 + 3/4) = 28.0
        fp6_max = 28.0
    else:
        raise ValueError(f"Unsupported FP6 format: {format}")
    
    sign = x.sign()
    x_abs = x.abs()
    
    # Calculate block-wise scale using E8M0 format (power of 2)
    scale = torch.pow(2.0, torch.floor(torch.log2(fp6_max / x_abs.max(dim=-1, keepdim=True)[0])))
    scale = torch.where((0 < scale) * (scale < torch.inf), scale, 1.0)
    
    # Quantize absolute values
    x_fp6_abs = fp6_quantize_positive(x_abs * scale, stochastic_rounding, format) / scale
    
    return sign * x_fp6_abs


def mxfp6_torch(x: torch.Tensor, 
                stochastic_rounding: bool = False, 
                scaled_value_format: str = 'e3m2') -> torch.Tensor:
    """
    MXFP6 quantization using PyTorch.
    
    Args:
        x: Input tensor
        stochastic_rounding: Whether to use stochastic rounding
        scaled_value_format: 'e2m3' or 'e3m2'
    
    Returns:
        Quantized tensor
    """
    if scaled_value_format not in ['e2m3', 'e3m2']:
        raise RuntimeError(f"do not support scaled_value_format {scaled_value_format}")
    
    return fp6_scaled(x, stochastic_rounding, scaled_value_format)


def mxfp8_torch(x:torch.Tensor, 
                   stochastic_rounding:bool=False, 
                   scaled_value_format:str='e4m3') -> torch.Tensor:
    if scaled_value_format == "e4m3":
        fp8_max = 448
    elif scaled_value_format == "e5m2":
        fp8_max = 57344
    else:
        raise RuntimeError(f"do not support scaled_value_format {scaled_value_format}")
    
    sign = x.sign()
    x_abs = x.abs()
    
    # Get max per row, clamp to avoid division by zero
    max_val = x_abs.max(dim=-1, keepdim=True)[0]
    max_val = torch.clamp(max_val, min=1e-12)  # Avoid division by zero
    
    # Calculate scale safely
    ratio = fp8_max / max_val
    log_ratio = torch.log2(ratio)
    # Clamp log values to avoid extreme scales
    log_ratio = torch.clamp(log_ratio, min=-20, max=20)
    scale = torch.pow(2.0, torch.floor(log_ratio))
    
    # Additional safety check
    scale = torch.where((0 < scale) * (scale < torch.inf) * ~torch.isnan(scale), scale, 1.0)
    
    if scaled_value_format == "e4m3":
        x_fp8_abs = (x_abs * scale).to(torch.float8_e4m3fn).to(x.dtype) / scale
    elif scaled_value_format == "e5m2":
        x_fp8_abs = (x_abs * scale).to(torch.float8_e5m2).to(x.dtype) / scale
    
    result = sign * x_fp8_abs
    # Final safety check to replace any NaN/Inf with zeros
    result = torch.where(torch.isnan(result) | torch.isinf(result), torch.zeros_like(result), result)
    
    return result.to(x.dtype)


if __name__ == '__main__':

    device = torch.device('cpu')

    t = torch.randn([2, 32]).to(device)
    t_q = mxfp8_torch(t, stochastic_rounding=True)
    print(t_q)
