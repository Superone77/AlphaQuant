from __future__ import annotations
from dataclasses import dataclass
import torch
from .base import Quantizer, QuantizerConfig
from typing import Any, Dict, Optional
from .kernel.int_torch import fake_quant_fp4


@dataclass
class FP4Config(QuantizerConfig):
    def __init__(self, wq: Optional[str] = None, aq: Optional[str] = None, group_size: Optional[int] = 128, 
                 extra: Optional[Dict[str, Any]] = None, dtype="bfloat16", format: str = "e2m1", 
                 use_group_quant: bool = False):
        super().__init__(name="fp4", wq=wq, aq=aq, group_size=group_size, extra=extra, dtype=dtype)
        self.format = format
        self.use_group_quant = use_group_quant


class FP4Quantizer(Quantizer):
    """
    Standard FP4 quantizer supporting E2M1 format.
    """
    def __init__(self, cfg: FP4Config):
        super().__init__(cfg)
        self.format = cfg.format
        self.use_group_quant = getattr(cfg, 'use_group_quant', False)
        self.group_size = getattr(cfg, 'group_size', 128)

    @staticmethod
    def _calc_scale_max(max_val: torch.Tensor, qmax: float) -> torch.Tensor:
        return (max_val.clamp(min=1e-8)) / qmax

    def quantize_weight(self, w: torch.Tensor):
        """Quantize weights using FP4."""
        if not self.use_group_quant:
            # Per-tensor quantization
            w_deq = fake_quant_fp4(w, stochastic_rounding=False, format=self.format)
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
        
        # Get FP4 max value
        if self.format == "e2m1":
            fp4_max = 6.0
        else:
            raise ValueError(f"Unsupported FP4 format: {self.format}")
        
        # Quantize each group
        w_deq_groups = []
        for i in range(num_groups):
            group = w_grouped[i]
            
            # Calculate scale for this group
            maxv = group.abs().max()
            scale = self._calc_scale_max(maxv, fp4_max)
            
            # Scale, quantize, and dequantize
            group_scaled = group / scale
            group_quant = fake_quant_fp4(group_scaled, stochastic_rounding=False, format=self.format)
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
        if self.format == "e2m1":
            fp4_max = 6.0
        else:
            raise ValueError(f"Unsupported FP4 format: {self.format}")
        
        self.scale = self._calc_scale_max(maxv, fp4_max)
        self.zero = None

    def quantize_activation(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize activations using FP4."""
        if self.scale is None:
            # Dynamic quantization
            x_deq = fake_quant_fp4(x, stochastic_rounding=False, format=self.format)
            return x_deq
        else:
            # Static quantization using precomputed scale
            x_scaled = x * self.scale
            x_quant = fake_quant_fp4(x_scaled, stochastic_rounding=False, format=self.format)
            x_deq = x_quant / self.scale
            return x_deq
