from __future__ import annotations
from dataclasses import dataclass
import torch
from .base import Quantizer, QuantizerConfig
from typing import Any, Dict, Optional
from .kernel.int_torch import fake_quant_fp4


@dataclass
class FP4Config(QuantizerConfig):
    def __init__(self, wq: Optional[str] = None, aq: Optional[str] = None, group_size: Optional[int] = None, 
                 extra: Optional[Dict[str, Any]] = None, dtype="bfloat16", format: str = "e2m1"):
        super().__init__(name="fp4", wq=wq, aq=aq, group_size=group_size, extra=extra, dtype=dtype)
        self.format = format


class FP4Quantizer(Quantizer):
    """
    Standard FP4 quantizer supporting E2M1 format.
    """
    def __init__(self, cfg: FP4Config):
        super().__init__(cfg)
        self.format = cfg.format

    @staticmethod
    def _calc_scale_max(max_val: torch.Tensor, qmax: float) -> torch.Tensor:
        return (max_val.clamp(min=1e-8)) / qmax

    def quantize_weight(self, w: torch.Tensor):
        """Quantize weights using FP4."""
        w_deq = fake_quant_fp4(w, stochastic_rounding=False, format=self.format)
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
