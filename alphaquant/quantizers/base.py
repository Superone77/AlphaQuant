from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import torch


@dataclass
class QuantSchemeConfig:
    """Base quantization scheme configuration container.

    This is intentionally minimal and can be extended with scheme-specific fields.
    """

    name: str
    wq: Optional[str] = None  # weight quant scheme
    aq: Optional[str] = None  # activation quant scheme
    group_size: Optional[int] = None
    extra: Optional[Dict[str, Any]] = None

    @staticmethod
    def from_dict(config: Dict[str, Any], default_name: str = "custom") -> "QuantSchemeConfig":
        name = config.get("name") or config.get("scheme") or default_name
        wq = config.get("wq")
        aq = config.get("aq")
        group_size = config.get("group") or config.get("group_size")
        extra = {k: v for k, v in config.items() if k not in {"name", "scheme", "wq", "aq", "group", "group_size", "pattern"}}
        return QuantSchemeConfig(name=name, wq=wq, aq=aq, group_size=group_size, extra=extra or None)


@dataclass
class QuantizerConfig:
    group_size: int = 64         # elements per scale (for weights); for activations use channel-first flatten groups
    per_channel: bool = True     # if True for weights: per-output-channel then groupwise inside channel
    symmetric: bool = True
    dtype: str = "float16"       # compute dtype after dequant
    fake: bool = True            # if True, do fake-quant (quant->dequant on the fly)

class Observer:
    def reset(self):
        raise NotImplementedError
    def observe(self, x: torch.Tensor):
        raise NotImplementedError
    def get_range(self) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

class Quantizer:
    def __init__(self, cfg: QuantizerConfig):
        self.cfg = cfg
        self.scale: Optional[torch.Tensor] = None
        self.zero: Optional[torch.Tensor] = None
        self.ready: bool = False
    # ---- interface ----
    def attach_observer(self, obs: Observer):
        self.obs = obs
        return self
    def reset(self):
        self.scale, self.zero, self.ready = None, None, False
        if hasattr(self, 'obs'):
            self.obs.reset()
    def calibrate_finish(self):
        """Use attached observer to compute scales for activations."""
        assert hasattr(self, 'obs'), "Observer not attached"
        lo, hi = self.obs.get_range()  # tensors broadcastable to input shape
        self._from_range(lo, hi)
        self.ready = True
    # ---- implement in subclass ----
    def quantize_weight(self, w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (q_w_int, scale). Optionally set internal fields. Implement per subclass."""
        raise NotImplementedError
    def quantize_activation(self, x: torch.Tensor) -> torch.Tensor:
        """Fake quantize activations using precomputed scales (observer)."""
        raise NotImplementedError
    def _from_range(self, lo: torch.Tensor, hi: torch.Tensor):
        raise NotImplementedError