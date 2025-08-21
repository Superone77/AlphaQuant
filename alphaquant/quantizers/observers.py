
from __future__ import annotations
import torch
from .base import Observer

class MinMaxObserver(Observer):
    def __init__(self, momentum: float = 1.0, eps: float = 1e-8):
        self.m = momentum
        self.eps = eps
        self.reset()
    def reset(self):
        self.min_val = None
        self.max_val = None
    def observe(self, x: torch.Tensor):
        x = x.detach()
        cur_min = x.amin(dim=0, keepdim=True)
        cur_max = x.amax(dim=0, keepdim=True)
        if self.min_val is None:
            self.min_val = cur_min
            self.max_val = cur_max
        else:
            self.min_val = self.m * self.min_val + (1 - self.m) * cur_min
            self.max_val = self.m * self.max_val + (1 - self.m) * cur_max
    def get_range(self):
        assert self.min_val is not None and self.max_val is not None, "Observer empty"
        return self.min_val, self.max_val