from __future__ import annotations
from typing import Iterable, Dict, Any, List, Tuple
import re
import torch.nn as nn
from ..modules.quant_linear import QuantLinear, QuantLinearConfig
import json
import fnmatch
import torch
from alphaquant.modules.quant_linear import QuantLinearConfig
from alphaquant.quantizers.mxfp4 import MXFP4Quantizer, MXFP4Config
from alphaquant.quantizers.mxfp8 import MXFP8Quantizer, MXFP8Config
from alphaquant.quantizers.fp4 import FP4Quantizer, FP4Config
from alphaquant.quantizers.fp8 import FP8Quantizer, FP8Config
from alphaquant.quantizers.int_quantizers import (
    INT2Quantizer, INT2Config,
    INT4Quantizer, INT4Config,
    INT6Quantizer, INT6Config,
    INT8Quantizer, INT8Config
)
from alphaquant.quantizers.base import NoQuantizer, NoQuantConfig


def _match(name: str, include: Iterable[str], exclude: Iterable[str]) -> bool:
    inc_ok = True if not include else any((k in name) or re.fullmatch(k, name) for k in include)
    exc_ok = not any((k in name) or re.fullmatch(k, name) for k in exclude)
    return inc_ok and exc_ok


def load_layer_config(json_path: str) -> Dict[str, Any]:
    """Load layer-wise quantization configuration from a JSON file.

    Expected schema (example):
    {
      "default": {"wq": "mxfp8", "aq": "mxfp8", "group_size": 128},
      "overrides": [
        {"pattern": "model.layers.0.*", "wq": "mxfp4"},
        {"pattern": "model.layers.*.q_proj", "wq": "mxfp4", "aq": "mxfp8", "group_size": 64}
      ]
    }
    - pattern: glob-style pattern matched against module names from model.named_modules()
    - default: global scheme fields applied before overrides
    """
    with open(json_path, "r") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Layer config JSON must be a JSON object at top-level")
    if "default" in cfg and not isinstance(cfg["default"], dict):
        raise ValueError("'default' must be a JSON object")
    if "overrides" in cfg and not isinstance(cfg["overrides"], list):
        raise ValueError("'overrides' must be a JSON array")
    for ov in cfg.get("overrides", []):
        if not isinstance(ov, dict) or "pattern" not in ov:
            raise ValueError("Each override must be an object with a 'pattern' key")
    return cfg


def resolve_scheme_for_module(module_name: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve the quantization scheme dict for a module name using defaults + overrides.

    Returns a flat dict of fields like {wq, aq, group_size, ...}.
    """
    resolved = dict(cfg.get("default", {}))
    for override in cfg.get("overrides", []):
        pattern = override.get("pattern")
        if pattern and fnmatch.fnmatch(module_name, pattern):
            for key, value in override.items():
                if key == "pattern":
                    continue
                resolved[key] = value
    return resolved


def plan_model_layer_schemes(model: Any, cfg: Dict[str, Any], target_module_classes: Iterable[str] = ("Linear", "QuantLinear")) -> List[Tuple[str, Dict[str, Any]]]:
    """Create a plan for applying layer-wise schemes across the model.

    Only modules whose class names are in `target_module_classes` are considered.
    Returns a list of (module_name, scheme_dict).
    """
    plans: List[Tuple[str, Dict[str, Any]]] = []
    for module_name, module in model.named_modules():
        cls_name = module.__class__.__name__
        if cls_name in target_module_classes:
            scheme = resolve_scheme_for_module(module_name, cfg)
            if scheme.get("skip"):
                continue
            plans.append((module_name, scheme))
    return plans


def summarize_config(cfg: Dict[str, Any]) -> str:
    """Produce a short human-readable summary of a layer config JSON for logging."""
    default = cfg.get("default", {})
    overrides = cfg.get("overrides", [])
    lines = [
        "Layer-wise config:",
        f"  default: {default}",
        f"  overrides ({len(overrides)}):",
    ]
    for idx, ov in enumerate(overrides):
        lines.append(f"    {idx:02d}: pattern={ov.get('pattern')} fields={{" + ", ".join([f"{k}={v}" for k, v in ov.items() if k != 'pattern']) + "}}")
    return "\n".join(lines)


def load_quantization_plan(plan_path: str) -> Dict[str, Dict[str, Any]]:
    """Load quantization plan from JSON file."""
    with open(plan_path, 'r') as f:
        plan = json.load(f)
    return plan


def create_quantizer_from_scheme(scheme: Dict[str, Any], dtype: str) -> Tuple[Any, Any]:
    """Create quantizer class and config from scheme dict."""
    wq_scheme = scheme.get('wq', 'bf16')
    aq_scheme = scheme.get('aq', 'bf16')
    group_size = scheme.get('group_size', 32)
    
    # Map scheme names to quantizer classes
    quantizer_map = {
        'mxfp4': (MXFP4Quantizer, MXFP4Config),
        'mxfp8': (MXFP8Quantizer, MXFP8Config),
        'fp4': (FP4Quantizer, FP4Config),
        'fp8': (FP8Quantizer, FP8Config),
        'int2': (INT2Quantizer, INT2Config),
        'int4': (INT4Quantizer, INT4Config),
        'int6': (INT6Quantizer, INT6Config),
        'int8': (INT8Quantizer, INT8Config),
        'bf16': (NoQuantizer, NoQuantConfig)
    }
    
    if wq_scheme not in quantizer_map:
        raise ValueError(f"Unknown weight quantization scheme: {wq_scheme}")
    if aq_scheme not in quantizer_map:
        raise ValueError(f"Unknown activation quantization scheme: {aq_scheme}")
    
    WQ, WCfg = quantizer_map[wq_scheme]
    AQ, ACfg = quantizer_map[aq_scheme]
    
    return (WQ, WCfg), (AQ, ACfg)


def apply_layer_wise_quantization(model: Any, plan: Dict[str, Dict[str, Any]], dtype: str) -> List[str]:
    """Apply layer-wise quantization based on the plan."""
    replaced_modules = []
    
    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and module_name in plan:
            scheme = plan[module_name]
            
            # Create quantizer configs
            (WQ, WCfg), (AQ, ACfg) = create_quantizer_from_scheme(scheme, dtype)
            
            # Create quantization config
            qcfg = QuantLinearConfig(
                weight_quantizer_cls=WQ,
                weight_quantizer_cfg=WCfg(group_size=scheme.get('group_size', 32), dtype=dtype),
                act_quantizer_cls=AQ,
                act_quantizer_cfg=ACfg(group_size=scheme.get('group_size', 32), dtype=dtype),
                bias=True
            )
            
            # Replace the module
            parent_name = module_name.rsplit('.', 1)[0] if '.' in module_name else ''
            attr = module_name.split('.')[-1]
            parent = model.get_submodule(parent_name) if parent_name else model
            
            # Create QuantLinear replacement
            from alphaquant.modules.quant_linear import QuantLinear
            qlin = QuantLinear(module.in_features, module.out_features, bias=module.bias is not None, qcfg = qcfg)
            if module.bias is not None:
                qlin.inner.bias.data = module.bias.data.clone()
            qlin.inner.weight.data = module.weight.data.clone()
            
            setattr(parent, attr, qlin)
            # print(f"Finished Quantization for {module_name} with {scheme} ")
            replaced_modules.append(module_name)
    
    return replaced_modules