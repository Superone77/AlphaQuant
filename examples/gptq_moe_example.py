#!/usr/bin/env python3
"""
Example of GPTQ quantization for MoE models with routing-aware optimization.

This demonstrates:
1. Automatic MoE architecture detection
2. Routing-score-weighted Hessian computation
3. Expert utilization tracking
4. Mixed-precision configuration for MoE
"""

import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from alphaquant.gptq import (
    GPTQMoE,
    MoEGPTQContext,
    detect_moe_architecture,
    create_gptq_for_layer
)
from alphaquant.gptq.model_utils import find_layers, get_layers_for_model
from alphaquant.quantizers.mxfp4 import MXFP4Quantizer, MXFP4Config


def main():
    """Main demonstration function."""
    
    print("=" * 80)
    print("AlphaQuant MoE-Enhanced GPTQ Example")
    print("=" * 80)
    
    # ===== Configuration =====
    MODEL_NAME = "allenai/OLMoE-1B-7B-0924"  # Example MoE model
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # ===== Step 1: Load Model =====
    print("\n[Step 1] Loading MoE model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map='cpu',
        trust_remote_code=True
    )
    model.eval()
    
    # ===== Step 2: Detect MoE Architecture =====
    print("\n[Step 2] Detecting MoE architecture...")
    arch_type = detect_moe_architecture(model)
    print(f"Detected architecture: {arch_type}")
    
    # ===== Step 3: Setup MoE Context =====
    print("\n[Step 3] Setting up MoE routing context...")
    moe_context = MoEGPTQContext(model_type=arch_type)
    
    # ===== Step 4: Example Layer Quantization =====
    print("\n[Step 4] Demonstrating expert layer quantization...")
    
    layers = get_layers_for_model(model, model_type='auto')
    
    # Take first layer as example
    layer_idx = 0
    layer = layers[layer_idx].to(DEVICE)
    
    # Find expert layers
    all_layers = find_layers(layer, layers=[torch.nn.Linear])
    expert_layers = {
        name: module for name, module in all_layers.items()
        if 'expert' in name.lower()
    }
    
    print(f"\nFound {len(expert_layers)} expert layers in layer {layer_idx}")
    
    if expert_layers:
        # Pick first expert as demonstration
        expert_name, expert_layer = list(expert_layers.items())[0]
        print(f"\nQuantizing expert layer: {expert_name}")
        
        # Extract expert ID from name (e.g., "mlp.experts.0.up_proj" -> 0)
        expert_id = None
        if 'experts.' in expert_name:
            try:
                parts = expert_name.split('experts.')[1].split('.')
                expert_id = int(parts[0])
            except:
                pass
        
        # Create MoE-enhanced GPTQ
        gptq_moe = GPTQMoE(expert_layer, expert_id=expert_id)
        
        print(f"  Expert ID: {expert_id}")
        print(f"  Weight shape: {expert_layer.weight.shape}")
        
        # ===== Step 5: Simulate Calibration =====
        print("\n[Step 5] Simulating calibration with routing...")
        
        # Simulate some calibration data
        nsamples = 8
        batch_size = 4
        seq_len = 128
        hidden_size = expert_layer.in_features
        num_experts = 8  # Example for OLMoE
        top_k = 2
        
        for sample_idx in range(nsamples):
            # Simulate expert input
            inp = torch.randn(batch_size * seq_len, hidden_size).to(DEVICE)
            
            # Simulate routing scores and expert selection
            # In real case, these come from the router
            routing_scores = torch.rand(batch_size, seq_len, top_k).to(DEVICE)
            routing_scores = routing_scores / routing_scores.sum(dim=-1, keepdim=True)
            
            selected_experts = torch.randint(0, num_experts, (batch_size, seq_len, top_k)).to(DEVICE)
            # Make sure our expert is selected at least sometimes
            selected_experts[0, 0, 0] = expert_id if expert_id is not None else 0
            
            # Add batch with routing information
            if expert_id is not None:
                gptq_moe.add_batch_with_routing(
                    inp=inp,
                    routing_scores=routing_scores,
                    selected_experts=selected_experts,
                    expert_num=expert_id,
                    num_experts=num_experts
                )
        
        print(f"  Collected {gptq_moe.nsamples} samples")
        print(f"  Expert utilization: {gptq_moe.utilization_count} activations")
        print(f"  Hessian shape: {gptq_moe.H.shape}")
        
        # ===== Step 6: Quantize =====
        print("\n[Step 6] Applying GPTQ quantization...")
        
        # Create quantizer (MXFP4 for experts as per best practices)
        quantizer = MXFP4Quantizer(MXFP4Config(group_size=32, dtype='bfloat16'))
        
        # Apply GPTQ
        losses = gptq_moe.fasterquant(
            quantizer=quantizer,
            blocksize=128,
            percdamp=0.01,
            groupsize=32
        )
        
        print(f"  Quantization complete")
        print(f"  Average loss: {losses.mean().item():.6f}")
        print(f"  Max loss: {losses.max().item():.6f}")
    
    # ===== Summary =====
    print("\n" + "=" * 80)
    print("MoE GPTQ Summary")
    print("=" * 80)
    print(f"Model: {MODEL_NAME}")
    print(f"Architecture: {arch_type}")
    print(f"MoE-specific features:")
    print("  ✓ Routing-score-weighted Hessian")
    print("  ✓ Expert utilization tracking")
    print("  ✓ Per-expert quantization")
    print("\nKey differences from standard GPTQ:")
    print("  1. Hessian weighted by routing probabilities")
    print("  2. Accounts for sparse expert activation")
    print("  3. Can track which experts are used more")
    print("=" * 80)


if __name__ == '__main__':
    main()

