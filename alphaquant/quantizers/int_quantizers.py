from __future__ import annotations
from dataclasses import dataclass
import torch
from .base import Quantizer, QuantizerConfig
from typing import Any, Dict, Optional
from .kernel.int_torch import (
    fake_quant_int2, fake_quant_int3, fake_quant_int4, fake_quant_int6, fake_quant_int8
)


# ============================================================================
# Configuration Classes
# ============================================================================

@dataclass
class INT2Config(QuantizerConfig):
    def __init__(self, wq: Optional[str] = None, aq: Optional[str] = None, group_size: Optional[int] = 128, 
                 extra: Optional[Dict[str, Any]] = None, dtype="bfloat16", symmetric: bool = True, 
                 use_group_quant: bool = False):
        super().__init__(name="int2", wq=wq, aq=aq, group_size=group_size, extra=extra, dtype=dtype)
        self.symmetric = symmetric
        self.use_group_quant = use_group_quant


@dataclass
class INT3Config(QuantizerConfig):
    def __init__(self, wq: Optional[str] = None, aq: Optional[str] = None, group_size: Optional[int] = 128, 
                 extra: Optional[Dict[str, Any]] = None, dtype="bfloat16", symmetric: bool = True, 
                 use_group_quant: bool = False):
        super().__init__(name="int3", wq=wq, aq=aq, group_size=group_size, extra=extra, dtype=dtype)
        self.symmetric = symmetric
        self.use_group_quant = use_group_quant


@dataclass
class INT4Config(QuantizerConfig):
    def __init__(self, wq: Optional[str] = None, aq: Optional[str] = None, group_size: Optional[int] = 128, 
                 extra: Optional[Dict[str, Any]] = None, dtype="bfloat16", symmetric: bool = True, 
                 use_group_quant: bool = False):
        super().__init__(name="int4", wq=wq, aq=aq, group_size=group_size, extra=extra, dtype=dtype)
        self.symmetric = symmetric
        self.use_group_quant = use_group_quant


@dataclass
class INT6Config(QuantizerConfig):
    def __init__(self, wq: Optional[str] = None, aq: Optional[str] = None, group_size: Optional[int] = 128, 
                 extra: Optional[Dict[str, Any]] = None, dtype="bfloat16", symmetric: bool = True, 
                 use_group_quant: bool = False):
        super().__init__(name="int6", wq=wq, aq=aq, group_size=group_size, extra=extra, dtype=dtype)
        self.symmetric = symmetric
        self.use_group_quant = use_group_quant


@dataclass
class INT8Config(QuantizerConfig):
    def __init__(self, wq: Optional[str] = None, aq: Optional[str] = None, group_size: Optional[int] = 128, 
                 extra: Optional[Dict[str, Any]] = None, dtype="bfloat16", symmetric: bool = True, 
                 use_group_quant: bool = False):
        super().__init__(name="int8", wq=wq, aq=aq, group_size=group_size, extra=extra, dtype=dtype)
        self.symmetric = symmetric
        self.use_group_quant = use_group_quant


# ============================================================================
# Base Integer Quantizer Class
# ============================================================================

class BaseIntQuantizer(Quantizer):
    """Base class for integer quantizers with common functionality."""
    
    def __init__(self, cfg: QuantizerConfig, qmin: int, qmax: int):
        super().__init__(cfg)
        self.qmin = qmin
        self.qmax = qmax
        self.symmetric = getattr(cfg, 'symmetric', True)
        self.use_group_quant = getattr(cfg, 'use_group_quant', False)
        self.group_size = getattr(cfg, 'group_size', 128)

    @staticmethod
    def _calc_scale_max(max_val: torch.Tensor, qmax: int) -> torch.Tensor:
        return (max_val.clamp(min=1e-8)) / qmax

    def _get_quantization_params(self, x_min: torch.Tensor, x_max: torch.Tensor):
        """Calculate scale and zero point for quantization."""
        if self.symmetric:
            maxv = torch.maximum(x_max.abs(), x_min.abs())
            scale = self._calc_scale_max(maxv, self.qmax)
            zero_point = torch.zeros_like(scale)
        else:
            scale = (x_max - x_min) / (self.qmax - self.qmin)
            zero_point = self.qmin - x_min / scale
        return scale, zero_point

    def _quantize_with_params(self, x: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor):
        """Quantize tensor using precomputed scale and zero point."""
        if self.symmetric:
            x_scaled = x / scale
            x_quant = self._quantize_kernel(x_scaled, stochastic_rounding=False, symmetric=True)
            x_deq = x_quant * scale
        else:
            x_scaled = x / scale + zero_point
            x_quant = self._quantize_kernel(x_scaled, stochastic_rounding=False, symmetric=False)
            x_deq = (x_quant - zero_point) * scale
        return x_deq

    def _quantize_weight_group(self, w: torch.Tensor):
        """Quantize weights with group quantization."""
        original_shape = w.shape
        w_flat = w.flatten()
        
        # Calculate number of groups
        numel = w_flat.numel()
        num_groups = (numel + self.group_size - 1) // self.group_size
        
        # Pad to multiple of group_size
        pad_size = num_groups * self.group_size - numel
        if pad_size > 0:
            w_flat = torch.cat([w_flat, torch.zeros(pad_size, device=w.device, dtype=w.dtype)])
        
        # Reshape to [num_groups, group_size]
        w_grouped = w_flat.reshape(num_groups, self.group_size)
        
        # Quantize each group
        w_deq_groups = []
        for i in range(num_groups):
            group = w_grouped[i]
            
            # Calculate scale and zero point for this group
            if self.symmetric:
                maxv = group.abs().max()
                scale = self._calc_scale_max(maxv, self.qmax)
                zero_point = torch.zeros_like(scale)
            else:
                x_min = group.min()
                x_max = group.max()
                scale = (x_max - x_min) / (self.qmax - self.qmin)
                zero_point = self.qmin - x_min / scale
            
            # Quantize and dequantize
            group_deq = self._quantize_with_params(group, scale, zero_point)
            w_deq_groups.append(group_deq)
        
        # Concatenate and remove padding
        w_deq_flat = torch.cat(w_deq_groups)
        if pad_size > 0:
            w_deq_flat = w_deq_flat[:-pad_size]
        
        # Reshape back to original shape
        w_deq = w_deq_flat.reshape(original_shape)
        return w_deq

    def quantize_weight(self, w: torch.Tensor):
        """Quantize weights using the appropriate integer quantization kernel."""
        if not self.use_group_quant:
            # Per-tensor quantization
            return self._quantize_kernel(w, stochastic_rounding=False, symmetric=self.symmetric)
        else:
            # Group quantization
            return self._quantize_weight_group(w)

    def _from_range(self, lo: torch.Tensor, hi: torch.Tensor):
        """Calculate scale from range for activation quantization."""
        scale, zero_point = self._get_quantization_params(lo, hi)
        self.scale = scale
        self.zero = zero_point

    def quantize_activation(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize activations using integer quantization."""
        if self.scale is None:
            # Dynamic quantization
            return self._quantize_kernel(x, stochastic_rounding=False, symmetric=self.symmetric)
        else:
            # Static quantization using precomputed scale
            return self._quantize_with_params(x, self.scale, self.zero)


# ============================================================================
# Specific Integer Quantizer Classes
# ============================================================================

class INT2Quantizer(BaseIntQuantizer):
    """INT2 quantizer (2-bit integer quantization)."""
    
    def __init__(self, cfg: INT2Config):
        super().__init__(cfg, qmin=-2 if cfg.symmetric else 0, qmax=1 if cfg.symmetric else 3)
    
    def _quantize_kernel(self, x: torch.Tensor, stochastic_rounding: bool = False, symmetric: bool = True):
        return fake_quant_int2(x, stochastic_rounding, symmetric)


class INT3Quantizer(BaseIntQuantizer):
    """INT3 quantizer (3-bit integer quantization)."""
    
    def __init__(self, cfg: INT3Config):
        super().__init__(cfg, qmin=-4 if cfg.symmetric else 0, qmax=3 if cfg.symmetric else 7)
    
    def _quantize_kernel(self, x: torch.Tensor, stochastic_rounding: bool = False, symmetric: bool = True):
        return fake_quant_int3(x, stochastic_rounding, symmetric)


class INT4Quantizer(BaseIntQuantizer):
    """INT4 quantizer (4-bit integer quantization)."""
    
    def __init__(self, cfg: INT4Config):
        super().__init__(cfg, qmin=-8 if cfg.symmetric else 0, qmax=7 if cfg.symmetric else 15)
    
    def _quantize_kernel(self, x: torch.Tensor, stochastic_rounding: bool = False, symmetric: bool = True):
        return fake_quant_int4(x, stochastic_rounding, symmetric)


class INT6Quantizer(BaseIntQuantizer):
    """INT6 quantizer (6-bit integer quantization)."""
    
    def __init__(self, cfg: INT6Config):
        super().__init__(cfg, qmin=-32 if cfg.symmetric else 0, qmax=31 if cfg.symmetric else 63)
    
    def _quantize_kernel(self, x: torch.Tensor, stochastic_rounding: bool = False, symmetric: bool = True):
        return fake_quant_int6(x, stochastic_rounding, symmetric)


class INT8Quantizer(BaseIntQuantizer):
    """INT8 quantizer (8-bit integer quantization)."""
    
    def __init__(self, cfg: INT8Config):
        super().__init__(cfg, qmin=-128 if cfg.symmetric else 0, qmax=127 if cfg.symmetric else 255)
    
    def _quantize_kernel(self, x: torch.Tensor, stochastic_rounding: bool = False, symmetric: bool = True):
        return fake_quant_int8(x, stochastic_rounding, symmetric)
