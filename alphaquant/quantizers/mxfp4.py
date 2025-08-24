from __future__ import annotations
import os
from dataclasses import dataclass
import torch
from .base import Quantizer, QuantizerConfig
from typing import Any, Dict, Optional
from .kernel.fp_torch import fp4_121_scaled, fake_quant_mxfp4


@dataclass
class MXFP4Config(QuantizerConfig):
    def __init__(self, wq: Optional[str] = None, aq: Optional[str] = None, group_size: Optional[int] = None, extra: Optional[Dict[str, Any]] = None, dtype = "bfloat16"):
        super().__init__(name="mxfp4", wq=wq, aq=aq, group_size=group_size, extra=extra, dtype=dtype)

class MXFP4Quantizer(Quantizer):
    """
    Minimal stand-in for MXFP4: symmetric int4 with group-wise scale.
    We pack to int8 range [-8, 7] and store scale per group.
    """
    def __init__(self, cfg: MXFP4Config):
        super().__init__(cfg)

    @staticmethod
    def _calc_scale_max(max_val: torch.Tensor, qmax: int) -> torch.Tensor:
        scale = (max_val.clamp(min=1e-8)) / qmax
        return scale

    def _chunk_along(self, w: torch.Tensor, dim: int, size: int):
        n = w.size(dim)
        pad = (size - n % size) % size
        if pad:
            w = torch.nn.functional.pad(w, (0,0) if dim != -1 else (0,pad)) if dim == -1 else torch.nn.functional.pad(w, (0,0,0,pad))
        return w, pad

    def quantize_weight(self, w: torch.Tensor):
        w_deq = fp4_121_scaled(w,scale_format="e8m0")
        return w_deq

    def _from_range(self, lo: torch.Tensor, hi: torch.Tensor):
        maxv = torch.maximum(hi.abs(), lo.abs())
        self.scale = self._calc_scale_max(maxv, qmax=7)
        self.zero = None

    def quantize_activation(self, x: torch.Tensor) -> torch.Tensor:
        if self.scale == None:
            if os.environ['USE_TRITON_MXFP4'] == 1:
                x_deq = fake_quant_mxfp4(x)
            else:
                x_deq = fp4_121_scaled(x,scale_format="e8m0")
            return x_deq
        else:
            raise ValueError(f"do not support static quantization for activation")