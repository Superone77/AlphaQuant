from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import torch

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