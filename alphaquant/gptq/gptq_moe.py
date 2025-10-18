"""
MoE-enhanced GPTQ implementation.

This module extends the base GPTQ algorithm with MoE-specific optimizations:
- Routing score weighted Hessian computation
- Expert utilization tracking
- Shared expert handling
"""

import math
import time
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gptq import GPTQ, GPTQConfig


logger = logging.getLogger(__name__)


class GPTQMoE(GPTQ):
    """
    GPTQ quantizer for MoE expert layers.
    
    Extends base GPTQ with routing-score-weighted Hessian computation
    for better quantization of sparsely activated experts.
    """
    
    def __init__(self, layer: nn.Linear, expert_id: Optional[int] = None, use_hadamard: bool = False):
        """
        Initialize GPTQ for an MoE expert layer.
        
        Args:
            layer: The expert linear layer to quantize
            expert_id: ID of this expert (for tracking)
            use_hadamard: Whether to apply Hadamard transform for outlier suppression
        """
        super().__init__(layer, use_hadamard=use_hadamard)
        self.expert_id = expert_id
        self.utilization_count = 0  # Track how often this expert is used
    
    def add_batch_with_routing(
        self,
        inp: torch.Tensor,
        routing_scores: torch.Tensor,
        selected_experts: torch.Tensor,
        expert_num: int,
        num_experts: int,
        out: Optional[torch.Tensor] = None
    ):
        """
        Add a batch with routing score weighting for MoE experts.
        
        This computes a routing-weighted Hessian that accounts for:
        1. Which tokens are routed to this expert
        2. The routing probability/score for this expert
        
        Args:
            inp: Input tensor to the expert
            routing_scores: Routing probabilities [batch, seq, top_k]
            selected_experts: Selected expert indices [batch, seq, top_k]
            expert_num: This expert's ID
            num_experts: Total number of experts
            out: Output tensor (not used, for compatibility)
        """
        # Create one-hot mask for this expert
        # Shape: [num_experts, top_k, batch*seq]
        expert_mask = F.one_hot(
            selected_experts.view(-1, selected_experts.shape[-1]),
            num_classes=num_experts
        ).permute(2, 1, 0)
        
        # Find tokens routed to this expert
        idx, top_x = torch.where(expert_mask[expert_num])
        
        if len(idx) == 0:
            # This expert wasn't used in this batch
            return
        
        self.utilization_count += len(idx)
        
        # Get routing scores for this expert
        # Shape: [num_routed_tokens, 1]
        scores = routing_scores.view(-1, routing_scores.shape[-1])[top_x, idx, None].to(inp.device)
        
        # Weight inputs by sqrt of routing scores
        # This ensures Hessian H = X^T X is weighted by routing probability
        weighted_inp = inp * torch.sqrt(scores)
        
        # Reshape and transpose for Hessian computation
        if len(weighted_inp.shape) == 2:
            weighted_inp = weighted_inp.unsqueeze(0)
        tmp = weighted_inp.shape[0]
        if len(weighted_inp.shape) == 3:
            weighted_inp = weighted_inp.reshape((-1, weighted_inp.shape[-1]))
        
        weighted_inp = weighted_inp.t()
        
        # Update running Hessian
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        
        weighted_inp = math.sqrt(2 / self.nsamples) * weighted_inp.float()
        self.H += weighted_inp.matmul(weighted_inp.t())
    
    def add_batch_shared_expert(
        self,
        inp: torch.Tensor,
        routing_scores: torch.Tensor,
        out: Optional[torch.Tensor] = None
    ):
        """
        Add a batch for shared expert with routing score weighting.
        
        Shared experts in models like Qwen-MoE have their own gating.
        
        Args:
            inp: Input tensor
            routing_scores: Routing scores for the shared expert [batch, seq]
            out: Output tensor (not used)
        """
        # Weight by sqrt of routing scores
        scores = torch.sqrt(routing_scores).to(inp.device)
        weighted_inp = inp * scores
        
        if len(weighted_inp.shape) == 2:
            weighted_inp = weighted_inp.unsqueeze(0)
        tmp = weighted_inp.shape[0]
        if len(weighted_inp.shape) == 3:
            weighted_inp = weighted_inp.reshape((-1, weighted_inp.shape[-1]))
        
        weighted_inp = weighted_inp.t()
        
        # Update Hessian
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        
        weighted_inp = math.sqrt(2 / self.nsamples) * weighted_inp.float()
        self.H += weighted_inp.matmul(weighted_inp.t())


class MoEGPTQContext:
    """
    Context manager for collecting routing information during MoE quantization.
    
    This tracks routing scores and expert selection across forward passes
    to enable routing-weighted Hessian computation.
    """
    
    def __init__(self, model_type: str = 'auto'):
        """
        Initialize MoE context.
        
        Args:
            model_type: Model architecture ('mixtral', 'qwen', 'deepseek', 'auto')
        """
        self.model_type = model_type
        self.routing_scores = []
        self.selected_experts = []
        self.shared_routing_scores = []
        self.hooks = []
    
    def register_routing_hooks(self, layer: nn.Module, layer_idx: int):
        """
        Register hooks to capture routing decisions.
        
        Args:
            layer: Transformer layer module
            layer_idx: Layer index
        """
        self.routing_scores = []
        self.selected_experts = []
        self.shared_routing_scores = []
        
        if self.model_type == 'olmoe':
            # OLMoE: top-8 routing
            def save_olmoe_routing(module, inp, out):
                routing_score = F.softmax(out, dim=-1, dtype=torch.float)
                routing_score, selected = torch.topk(routing_score, k=8, dim=-1)
                routing_score = routing_score / routing_score.sum(dim=-1, keepdim=True)
                self.routing_scores.append(routing_score)
                self.selected_experts.append(selected)
            
            if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'gate'):
                handle = layer.mlp.gate.register_forward_hook(save_olmoe_routing)
                self.hooks.append(handle)
        
        elif self.model_type == 'mixtral':
            # Mixtral: top-2 routing
            def save_mixtral_routing(module, inp, out):
                routing_score = F.softmax(out, dim=-1, dtype=torch.float)
                routing_score, selected = torch.topk(routing_score, k=2, dim=-1)
                routing_score = routing_score / routing_score.sum(dim=-1, keepdim=True)
                self.routing_scores.append(routing_score)
                self.selected_experts.append(selected)
            
            if hasattr(layer, 'block_sparse_moe') and hasattr(layer.block_sparse_moe, 'gate'):
                handle = layer.block_sparse_moe.gate.register_forward_hook(save_mixtral_routing)
                self.hooks.append(handle)
        
        elif self.model_type == 'qwen':
            # Qwen-MoE: top-k routing + shared expert
            def save_qwen_routing(module, inp, out):
                routing_score = F.softmax(out, dim=-1, dtype=torch.float)
                routing_score, selected = torch.topk(routing_score, k=4, dim=-1)
                self.routing_scores.append(routing_score)
                self.selected_experts.append(selected)
            
            def save_shared_routing(module, inp, out):
                shared_score = F.sigmoid(out)
                self.shared_routing_scores.append(shared_score)
            
            if hasattr(layer, 'mlp'):
                if hasattr(layer.mlp, 'gate'):
                    handle = layer.mlp.gate.register_forward_hook(save_qwen_routing)
                    self.hooks.append(handle)
                if hasattr(layer.mlp, 'shared_expert_gate'):
                    handle = layer.mlp.shared_expert_gate.register_forward_hook(save_shared_routing)
                    self.hooks.append(handle)
        
        elif self.model_type == 'deepseek':
            # DeepSeek-MoE
            def save_deepseek_routing(module, inp, out):
                # DeepSeek returns (selected_experts, routing_scores)
                self.selected_experts.append(out[0])
                self.routing_scores.append(out[1])
            
            if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'gate'):
                handle = layer.mlp.gate.register_forward_hook(save_deepseek_routing)
                self.hooks.append(handle)
    
    def clear_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def get_routing_info(self) -> Dict[str, Any]:
        """
        Get collected routing information.
        
        Returns:
            Dictionary with routing_scores, selected_experts, shared_routing_scores
        """
        return {
            'routing_scores': torch.stack(self.routing_scores) if self.routing_scores else None,
            'selected_experts': torch.stack(self.selected_experts) if self.selected_experts else None,
            'shared_routing_scores': torch.stack(self.shared_routing_scores) if self.shared_routing_scores else None,
        }


def create_gptq_for_layer(
    layer: nn.Module,
    layer_name: str,
    is_expert: bool = False,
    expert_id: Optional[int] = None,
    use_hadamard: bool = False
) -> GPTQ:
    """
    Create appropriate GPTQ instance for a layer.
    
    Args:
        layer: The layer to quantize
        layer_name: Full layer name
        is_expert: Whether this is an MoE expert layer
        expert_id: Expert ID if applicable
        use_hadamard: Whether to apply Hadamard transform for outlier suppression
        
    Returns:
        GPTQ or GPTQMoE instance
    """
    if is_expert:
        return GPTQMoE(layer, expert_id=expert_id, use_hadamard=use_hadamard)
    else:
        return GPTQ(layer, use_hadamard=use_hadamard)


def detect_moe_architecture(model: nn.Module) -> str:
    """
    Detect MoE architecture type from model.
    
    Args:
        model: The model
        
    Returns:
        Architecture type: 'mixtral', 'qwen', 'deepseek', 'olmoe', or 'standard'
    """
    model_name = type(model).__name__.lower()
    
    if 'mixtral' in model_name:
        return 'mixtral'
    elif 'qwen' in model_name or 'qwen2moe' in model_name:
        return 'qwen'
    elif 'deepseek' in model_name:
        return 'deepseek'
    elif 'olmoe' in model_name or 'olmo' in model_name:
        return 'olmoe'
    
    # Check for MoE layers
    for name, module in model.named_modules():
        if 'expert' in name.lower() or 'moe' in name.lower():
            if 'mixtral' in name.lower():
                return 'mixtral'
            elif 'qwen' in name.lower():
                return 'qwen'
            elif 'deepseek' in name.lower():
                return 'deepseek'
            elif 'olmoe' in name.lower() or 'olmo' in name.lower():
                return 'olmoe'
            return 'generic_moe'
    
    return 'standard'

