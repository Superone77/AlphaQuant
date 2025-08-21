from __future__ import annotations
from typing import Iterable, Dict, Any, List, Tuple
import re
import torch.nn as nn
from ..modules.quant_linear import QuantLinear, QuantLinearConfig
import json
import fnmatch


def _match(name: str, include: Iterable[str], exclude: Iterable[str]) -> bool:
    inc_ok = True if not include else any((k in name) or re.fullmatch(k, name) for k in include)
    exc_ok = not any((k in name) or re.fullmatch(k, name) for k in exclude)
    return inc_ok and exc_ok


def replace_linear_with_quant(model: nn.Module,
                              qcfg: QuantLinearConfig,
                              include: Iterable[str] = (),
                              exclude: Iterable[str] = (),
                              dry_run: bool = False) -> List[str]:
    """
    Replace selected nn.Linear with QuantLinear.
    - include/exclude: substrings or regex patterns on module qualified names
    Returns list of replaced module names.
    """
    replaced = []
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and _match(name, include, exclude):
            parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
            attr = name.split('.')[-1]
            parent = model.get_submodule(parent_name) if parent_name else model
            qlin = QuantLinear(module, qcfg)
            if not dry_run:
                setattr(parent, attr, qlin)
            replaced.append(name)
    return replaced


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