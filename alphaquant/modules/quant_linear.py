from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Type

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover - torch may not be installed in all envs
    torch = None
    nn = None

from ..quantizers.base import Quantizer

@dataclass
class QuantLinearConfig:
    weight_quantizer_cls: Type[Quantizer]
    weight_quantizer_cfg: object
    act_quantizer_cls: Type[Quantizer]
    act_quantizer_cfg: object
    bias: bool = True

class QuantLinear(nn.Module if nn is not None else object):
    """Placeholder quantized linear layer.

    This is a stub: it mirrors `nn.Linear` shape but does not perform real quantization.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True, **kwargs):
        if nn is None:
            raise RuntimeError("PyTorch is required to use QuantLinear")
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        self.inner = nn.Linear(in_features, out_features, bias=bias)
        self.quant_kwargs = kwargs

    def forward(self, x):  # type: ignore[override]
        return self.inner(x)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.use_bias}, quant={self.quant_kwargs}"