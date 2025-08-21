from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Type
import torch
import torch.nn as nn
from ..quantizers.base import Quantizer

@dataclass
class QuantLinearConfig:
    weight_quantizer_cls: Type[Quantizer]
    weight_quantizer_cfg: object
    act_quantizer_cls: Type[Quantizer]
    act_quantizer_cfg: object
    bias: bool = True

class QuantLinear(nn.Module):
    def __init__(self, ref: nn.Linear, qcfg: QuantLinearConfig):
        super().__init__()
        self.in_features = ref.in_features
        self.out_features = ref.out_features
        self.bias_flag = qcfg.bias and (ref.bias is not None)
        # quantizers
        self.wq = qcfg.weight_quantizer_cls(qcfg.weight_quantizer_cfg)
        self.aq = qcfg.act_quantizer_cls(qcfg.act_quantizer_cfg)
        # copy bias
        if self.bias_flag:
            self.bias = nn.Parameter(ref.bias.detach().clone())
        else:
            self.register_parameter('bias', None)
        # quantize weights now (offline)
        W = ref.weight.detach().to(torch.float32)
        Wq, scale = self.wq.quantize_weight(W)
        self.register_buffer('qweight', Wq)
        self.register_buffer('wscale', scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.aq, 'scale') and self.aq.scale is not None:
            x = self.aq.quantize_activation(x)
        # Using dequantized weight path (keeps it simple)
        return torch.nn.functional.linear(x, self.qweight, self.bias)