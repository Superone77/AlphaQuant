from __future__ import annotations
from typing import Iterable, Callable, Optional
import torch
import torch.nn as nn
from ..modules.quant_linear import QuantLinear
from ..quantizers.observers import MinMaxObserver

class Calibrator:
    """
    Attach observers to all QuantLinear activation quantizers, run a few batches,
    then finalize to compute scales.
    """
    def __init__(self, model: nn.Module, make_observer: Optional[Callable[[], MinMaxObserver]] = None):
        self.model = model
        self.make_observer = make_observer or (lambda: MinMaxObserver())
        self.hooks = []

    def prepare(self):
        # register forward-pre hooks to observe inputs of QuantLinear
        for mod in self.model.modules():
            if isinstance(mod, QuantLinear):
                obs = self.make_observer()
                mod.aq.attach_observer(obs)
                def _hook(m, inp):
                    x = inp[0]
                    m.aq.obs.observe(x)
                h = mod.register_forward_pre_hook(_hook)
                self.hooks.append(h)

    @torch.no_grad()
    def collect(self, dataloader: Iterable, num_batches: int = 16, device: str = "cuda"):
        self.model.eval()
        self.prepare()
        it = iter(dataloader)
        for i in range(num_batches):
            try:
                batch = next(it)
            except StopIteration:
                break
            # batch can be dict with 'input_ids' and 'attention_mask'
            if isinstance(batch, dict):
                batch = {k: v.to(device) for k, v in batch.items()}
                _ = self.model(**batch)
            else:
                batch = batch.to(device)
                _ = self.model(batch)
        # finalize
        for mod in self.model.modules():
            if isinstance(mod, QuantLinear):
                mod.aq.calibrate_finish()
        for h in self.hooks:
            h.remove()
        self.hooks.clear()