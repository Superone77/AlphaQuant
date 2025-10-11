from __future__ import annotations
from dataclasses import dataclass
import torch
from .base import Quantizer, QuantizerConfig
from typing import Any, Dict, Optional
from .kernel.fp_torch import fp6_quantize_positive


@dataclass
class FP6Config(QuantizerConfig):
    def __init__(self, wq: Optional[str] = None, aq: Optional[str] = None, 
                 group_size: Optional[int] = 128, 
                 extra: Optional[Dict[str, Any]] = None, dtype="bfloat16", 
                 format: str = "e3m2", 
                 use_group_quant: bool = False):
        """
        FP6 quantizer configuration.
        
        Args:
            wq: Weight quantization setting
            aq: Activation quantization setting
            group_size: Group size for group quantization
            extra: Extra configuration
            dtype: Data type
            format: FP6 format - 'e2m3' (2 exp bits, 3 mantissa bits) or 
                   'e3m2' (3 exp bits, 2 mantissa bits). Default: 'e3m2'
            use_group_quant: Whether to use group quantization
        """
        super().__init__(name="fp6", wq=wq, aq=aq, group_size=group_size, extra=extra, dtype=dtype)
        if format not in ['e2m3', 'e3m2']:
            raise ValueError(f"Unsupported FP6 format: {format}. Must be 'e2m3' or 'e3m2'")
        self.format = format
        self.use_group_quant = use_group_quant


class FP6Quantizer(Quantizer):
    """
    Standard FP6 quantizer supporting E2M3 and E3M2 formats.
    
    FP6 formats:
    - E2M3: 2 exponent bits (bias=1), 3 mantissa bits, max value ≈ 7.5
    - E3M2: 3 exponent bits (bias=3), 2 mantissa bits, max value ≈ 28.0
    """
    def __init__(self, cfg: FP6Config):
        super().__init__(cfg)
        self.format = cfg.format
        self.use_group_quant = getattr(cfg, 'use_group_quant', False)
        self.group_size = getattr(cfg, 'group_size', 128)
        
        # Set max value based on format
        if self.format == 'e2m3':
            self.fp6_max = 7.5
        elif self.format == 'e3m2':
            self.fp6_max = 28.0
        else:
            raise ValueError(f"Unsupported FP6 format: {self.format}")

    @staticmethod
    def _calc_scale_max(max_val: torch.Tensor, qmax: float) -> torch.Tensor:
        return (max_val.clamp(min=1e-8)) / qmax

    def _fake_quant_fp6(self, x: torch.Tensor, stochastic_rounding: bool = False) -> torch.Tensor:
        """
        Fake quantize to FP6 format.
        
        Args:
            x: Input tensor
            stochastic_rounding: Whether to use stochastic rounding
            
        Returns:
            Quantized and dequantized tensor
        """
        # Calculate scale
        x_abs = x.abs()
        x_max = x_abs.max()
        scale = self.fp6_max / x_max.clamp(min=1e-8)
        
        # Scale to FP6 range
        x_scaled = x * scale
        x_sign = torch.sign(x_scaled)
        x_abs_scaled = x_scaled.abs()
        
        # Quantize absolute values
        x_quant_abs = fp6_quantize_positive(x_abs_scaled, stochastic_rounding, self.format)
        x_quant = x_sign * x_quant_abs
        
        # Dequantize
        x_dequant = x_quant / scale
        
        return x_dequant

    def quantize_weight(self, w: torch.Tensor):
        """Quantize weights using FP6."""
        if not self.use_group_quant:
            # Per-tensor quantization
            w_deq = self._fake_quant_fp6(w, stochastic_rounding=False)
            return w_deq
        else:
            # Group quantization
            return self._quantize_weight_group(w)
    
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
            
            # Calculate scale for this group
            maxv = group.abs().max()
            scale = self._calc_scale_max(maxv, self.fp6_max)
            
            # Scale to FP6 range
            group_scaled = group / scale
            group_sign = torch.sign(group_scaled)
            group_abs_scaled = group_scaled.abs()
            
            # Quantize absolute values
            group_quant_abs = fp6_quantize_positive(group_abs_scaled, stochastic_rounding=False, format=self.format)
            group_quant = group_sign * group_quant_abs
            
            # Dequantize
            group_deq = group_quant * scale
            w_deq_groups.append(group_deq)
        
        # Concatenate and remove padding
        w_deq_flat = torch.cat(w_deq_groups)
        if pad_size > 0:
            w_deq_flat = w_deq_flat[:-pad_size]
        
        # Reshape back to original shape
        w_deq = w_deq_flat.reshape(original_shape)
        return w_deq

    def _from_range(self, lo: torch.Tensor, hi: torch.Tensor):
        """Calculate scale from range for activation quantization."""
        maxv = torch.maximum(hi.abs(), lo.abs())
        self.scale = self._calc_scale_max(maxv, self.fp6_max)
        self.zero = None

    def quantize_activation(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize activations using FP6."""
        if self.scale is None:
            # Dynamic quantization
            x_deq = self._fake_quant_fp6(x, stochastic_rounding=False)
            return x_deq
        else:
            # Static quantization using precomputed scale
            x_scaled = x * self.scale
            x_sign = torch.sign(x_scaled)
            x_abs_scaled = x_scaled.abs()
            
            # Quantize absolute values
            x_quant_abs = fp6_quantize_positive(x_abs_scaled, stochastic_rounding=False, format=self.format)
            x_quant = x_sign * x_quant_abs
            
            # Dequantize
            x_deq = x_quant / self.scale
            return x_deq

