"""
GPTQ quantization module for AlphaQuant.

This module provides mixed-precision GPTQ quantization capabilities
inspired by MoEQuant but integrated with AlphaQuant's quantizer system.
"""

from .gptq import GPTQ, GPTQConfig
from .gptq_moe import GPTQMoE, MoEGPTQContext, create_gptq_for_layer, detect_moe_architecture
from .quantize import gptq_quantize_model, rtn_quantize_model

__all__ = [
    'GPTQ',
    'GPTQConfig',
    'GPTQMoE',
    'MoEGPTQContext',
    'create_gptq_for_layer',
    'detect_moe_architecture',
    'gptq_quantize_model',
    'rtn_quantize_model',
]
