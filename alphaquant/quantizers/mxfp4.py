from __future__ import annotations
from dataclasses import dataclass
import torch
from .base import Quantizer, QuantizerConfig
from typing import Any, Dict, Optional

from .base import QuantSchemeConfig


@dataclass
class MXFP4Config(QuantSchemeConfig):
    def __init__(self, wq: Optional[str] = None, aq: Optional[str] = None, group_size: Optional[int] = None, extra: Optional[Dict[str, Any]] = None):
        super().__init__(name="mxfp4", wq=wq, aq=aq, group_size=group_size, extra=extra)

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
        cfg = self.cfg
        device = w.device
        out_ch, in_ch = w.shape
        g = cfg.group_size
        w_g = w.view(out_ch, in_ch // g, g)
        maxv = w_g.abs().amax(dim=-1, keepdim=True)
        scale = self._calc_scale_max(maxv, qmax=7)
        qw = torch.clamp((w_g / scale).round_(), -8, 7).to(torch.int8)
        # store dequantized for compute (simple path)
        w_deq = (qw * scale).view_as(w).to(getattr(torch, cfg.dtype))
        return w_deq, scale.squeeze(-1)  # scale shape: [out_ch, in_ch//g]

    def _from_range(self, lo: torch.Tensor, hi: torch.Tensor):
        maxv = torch.maximum(hi.abs(), lo.abs())
        self.scale = self._calc_scale_max(maxv, qmax=7)
        self.zero = None

    def quantize_activation(self, x: torch.Tensor) -> torch.Tensor:
        # fake quant with broadcasting scale from observer (per-feature along last dim)
        assert self.scale is not None
        scale = self.scale
        # Bring scale to last-dim broadcast if necessary
        while scale.dim() < x.dim():
            scale = scale.unsqueeze(0)
        xq = torch.clamp((x / scale).round_(), -8, 7)
        return (xq * scale).to(getattr(torch, self.cfg.dtype))