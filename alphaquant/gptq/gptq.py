"""
Core GPTQ algorithm implementation.

Based on the GPTQ paper and MoEQuant implementation, but adapted for
AlphaQuant's quantizer interface and mixed-precision requirements.
"""

import math
import time
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass

import torch
import torch.nn as nn

# Disable TF32 for better numerical stability in GPTQ
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


@dataclass
class GPTQConfig:
    """Configuration for GPTQ quantization."""
    blocksize: int = 128
    percdamp: float = 0.01
    groupsize: int = -1
    actorder: bool = False
    static_groups: bool = False


class GPTQ:
    """
    GPTQ quantizer for a single linear layer.
    
    This implementation follows the GPTQ algorithm:
    1. Collect Hessian information during calibration
    2. Use the Hessian to minimize quantization error
    3. Support group-wise quantization and activation ordering
    """

    def __init__(self, layer: nn.Linear):
        """
        Initialize GPTQ for a layer.
        
        Args:
            layer: The linear layer to quantize
        """
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp: torch.Tensor, out: Optional[torch.Tensor] = None):
        """
        Add a batch of layer inputs to accumulate Hessian information.
        
        Args:
            inp: Input tensor to the layer (activations)
            out: Output tensor (not used, for API compatibility)
        """
        # Normalize input shape
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        
        # Transpose for Hessian computation
        inp = inp.t()
        
        # Update running Hessian approximation
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        
        # Normalize by sample count for numerical stability
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def fasterquant(
        self,
        quantizer: Any,
        blocksize: int = 128,
        percdamp: float = 0.01,
        groupsize: int = -1,
        actorder: bool = False,
        static_groups: bool = False
    ) -> torch.Tensor:
        """
        Apply GPTQ quantization to the layer's weight.
        
        Args:
            quantizer: The quantizer to use (from AlphaQuant's quantizer system)
            blocksize: Block size for processing columns
            percdamp: Dampening percentage for Hessian
            groupsize: Group size for group-wise quantization (-1 = per-channel)
            actorder: Whether to use activation ordering
            static_groups: Whether to use static grouping
            
        Returns:
            Quantization losses per weight element
        """
        W = self.layer.weight.data.clone()
        W = W.float()

        tick = time.time()

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        if static_groups:
            import copy
            groups = []
            for i in range(0, self.columns, groupsize):
                quantizer_copy = copy.deepcopy(quantizer)
                # Note: quantizer should have a method to calibrate on weights
                groups.append(quantizer_copy)

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if groupsize != -1:
                    if not static_groups:
                        if (i1 + i) % groupsize == 0:
                            # Recalibrate quantizer for this group
                            # This depends on the quantizer interface
                            pass
                    else:
                        idx = i1 + i
                        if actorder:
                            idx = perm[idx]
                        quantizer = groups[idx // groupsize]

                # Quantize using AlphaQuant's quantizer
                # The quantizer should support quantize method
                q = self._quantize_column(quantizer, w.unsqueeze(1)).flatten()
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        torch.cuda.synchronize()

        if actorder:
            Q = Q[:, invperm]

        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(
            self.layer.weight.data.dtype
        )
        
        if torch.any(torch.isnan(self.layer.weight.data)):
            logging.warning('NaN in weights after GPTQ')
            raise ValueError('NaN in weights')

        logging.debug(f'GPTQ time: {time.time() - tick:.2f}s')
        return Losses

    def _quantize_column(self, quantizer: Any, w: torch.Tensor) -> torch.Tensor:
        """
        Quantize a weight column using the provided quantizer.
        
        This method handles different quantizer interfaces from AlphaQuant.
        """
        # For AlphaQuant quantizers that support quantize_weight
        if hasattr(quantizer, 'quantize_weight'):
            result = quantizer.quantize_weight(w)
            # Handle different return types
            if isinstance(result, tuple):
                return result[0]  # Return quantized weights
            else:
                return result
        # Fallback for simple quantization
        elif hasattr(quantizer, 'quantize'):
            return quantizer.quantize(w)
        else:
            # No quantization available
            return w

    def free(self):
        """Free memory used by GPTQ."""
        self.H = None
        torch.cuda.empty_cache()

