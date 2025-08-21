from __future__ import annotations
from typing import Iterable, Dict, Any, List
import re
import torch.nn as nn
from ..modules.quant_linear import QuantLinear, QuantLinearConfig


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