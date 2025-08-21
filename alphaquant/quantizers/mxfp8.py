from __future__ import annotations
from dataclasses import dataclass
import torch
from .base import Quantizer, QuantizerConfig

@dataclass
class MXFP8Config(QuantizerConfig):
    group_size: int = 64
    dtype: str = "float16"

class MXFP8Quantizer(Quantizer):
    """Minimal stand-in for MXFP8: symmetric int8 with group-wise scale."""
    def __init__(self, cfg: MXFP8Config):
        super().__init__(cfg)

    @staticmethod
    def _calc_scale_max(max_val: torch.Tensor, qmax: int) -> torch.Tensor:
        return (max_val.clamp(min=1e-8)) / qmax

    def quantize_weight(self, w: torch.Tensor):
        cfg = self.cfg
        out_ch, in_ch = w.shape
        g = cfg.group_size
        w_g = w.view(out_ch, in_ch // g, g)
        maxv = w_g.abs().amax(dim=-1, keepdim=True)
        scale = self._calc_scale_max(maxv, qmax=127)
        qw = torch.clamp((w_g / scale).round_(), -128, 127).to(torch.int8)
        w_deq = (qw * scale).view_as(w).to(getattr(torch, cfg.dtype))
        return w_deq, scale.squeeze(-1)

    def _from_range(self, lo: torch.Tensor, hi: torch.Tensor):
        maxv = torch.maximum(hi.abs(), lo.abs())
        self.scale = self._calc_scale_max(maxv, qmax=127)
        self.zero = None

    def quantize_activation(self, x: torch.Tensor) -> torch.Tensor:
        assert self.scale is not None
        scale = self.scale
        while scale.dim() < x.dim():
            scale = scale.unsqueeze(0)
        xq = torch.clamp((x / scale).round_(), -128, 127)
        return (xq * scale).to(getattr(torch, self.cfg.dtype))