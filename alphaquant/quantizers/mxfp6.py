from __future__ import annotations
from dataclasses import dataclass
import torch
from .base import Quantizer, QuantizerConfig
from typing import Any, Dict, Optional

from .kernel.fp_torch import mxfp6_torch


@dataclass
class MXFP6Config(QuantizerConfig):
    def __init__(self, wq: Optional[str] = None, aq: Optional[str] = None, 
                 group_size: Optional[int] = None, extra: Optional[Dict[str, Any]] = None,
                 dtype="bfloat16", format: str = "e3m2"):
        """
        MXFP6 quantizer configuration.
        
        Args:
            wq: Weight quantization setting
            aq: Activation quantization setting
            group_size: Group size for quantization
            extra: Extra configuration
            dtype: Data type
            format: FP6 format - 'e2m3' (2 exp bits, 3 mantissa bits) or 
                   'e3m2' (3 exp bits, 2 mantissa bits). Default: 'e3m2'
        """
        super().__init__(name="mxfp6", wq=wq, aq=aq, group_size=group_size, extra=extra, dtype=dtype)
        if format not in ['e2m3', 'e3m2']:
            raise ValueError(f"Unsupported FP6 format: {format}. Must be 'e2m3' or 'e3m2'")
        self.format = format


class MXFP6Quantizer(Quantizer):
    """
    MXFP6 quantizer with microscaling (block-wise scaling).
    
    Supports two FP6 formats:
    - E2M3: 2 exponent bits (bias=1), 3 mantissa bits, max value ≈ 7.5
    - E3M2: 3 exponent bits (bias=3), 2 mantissa bits, max value ≈ 28.0 (default)
    """
    def __init__(self, cfg: MXFP6Config):
        super().__init__(cfg)
        self.format = cfg.format
        
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

    def quantize_weight(self, w: torch.Tensor):
        """
        Quantize weights using MXFP6.
        
        Args:
            w: Weight tensor
            
        Returns:
            Quantized and dequantized weight tensor
        """
        w_deq = mxfp6_torch(w, scaled_value_format=self.format)
        return w_deq

    def _from_range(self, lo: torch.Tensor, hi: torch.Tensor):
        """
        Calculate scale from range for activation quantization.
        
        Args:
            lo: Lower bound of activation range
            hi: Upper bound of activation range
        """
        maxv = torch.maximum(hi.abs(), lo.abs())
        self.scale = self._calc_scale_max(maxv, qmax=self.fp6_max)
        self.zero = None

    def quantize_activation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Quantize activations using MXFP6.
        
        Args:
            x: Activation tensor
            
        Returns:
            Quantized and dequantized activation tensor
        """
        if self.scale == None:
            if x.numel() == 0:
                return x
            x_deq = mxfp6_torch(x, scaled_value_format=self.format)
            return x_deq
        else:
            raise ValueError(f"do not support static quantization for activation")

