from __future__ import annotations
from dataclasses import dataclass
import torch
from .base import Quantizer, QuantizerConfig
from typing import Any, Dict, Optional

from .kernel.fp_torch import mxfp8_torch


@dataclass
class MXFP8Config(QuantizerConfig):
    def __init__(self, wq: Optional[str] = None, aq: Optional[str] = None, group_size: Optional[int] = None, extra: Optional[Dict[str, Any]] = None,dtype="bfloat16"):
        super().__init__(name="mxfp8", wq=wq, aq=aq, group_size=group_size, extra=extra, dtype=dtype)

class MXFP8Quantizer(Quantizer):
    """Minimal stand-in for MXFP8: symmetric int8 with group-wise scale."""
    def __init__(self, cfg: MXFP8Config):
        super().__init__(cfg)

    @staticmethod
    def _calc_scale_max(max_val: torch.Tensor, qmax: int) -> torch.Tensor:
        return (max_val.clamp(min=1e-8)) / qmax

    def quantize_weight(self, w: torch.Tensor):
        w_deq = mxfp8_torch(w, scaled_value_format = "e4m3")
        return w_deq

    def _from_range(self, lo: torch.Tensor, hi: torch.Tensor):
        maxv = torch.maximum(hi.abs(), lo.abs())
        self.scale = self._calc_scale_max(maxv, qmax=127)
        self.zero = None

    def quantize_activation(self, x: torch.Tensor) -> torch.Tensor:
        if self.scale == None:
            if x.numel() == 0:
                return x
            x_deq = mxfp8_torch(x, scaled_value_format = "e5m2")
            return x_deq
        else:
            raise ValueError(f"do not support static quantization for activation")
