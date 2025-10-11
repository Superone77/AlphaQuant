#!/usr/bin/env python3
"""
Example showing how to use MXFP8 quantization after NaN fix
"""

import torch
from alphaquant.quantizers.mxfp8 import MXFP8Config, MXFP8Quantizer

def main():
    print("="*80)
    print("MXFP8 Quantization Example (After NaN Fix)")
    print("="*80)
    
    # Create quantizer
    config = MXFP8Config(wq="e4m3", aq="e5m2")
    quantizer = MXFP8Quantizer(config)
    
    # Test cases
    test_cases = [
        ("Normal weight matrix", torch.randn(128, 512)),
        ("Small values", torch.randn(64, 256) * 0.01),
        ("Large values", torch.randn(64, 256) * 100.0),
        ("Mixed with zeros", torch.cat([torch.zeros(32, 256), torch.randn(32, 256)], dim=0)),
    ]
    
    print("\n--- Weight Quantization (E4M3) ---")
    for name, w in test_cases:
        w_q = quantizer.quantize_weight(w)
        
        # Check for NaN/Inf
        has_nan = torch.isnan(w_q).any().item()
        has_inf = torch.isinf(w_q).any().item()
        
        # Calculate error
        mse = ((w - w_q) ** 2).mean().item()
        max_error = (w - w_q).abs().max().item()
        
        print(f"\n{name}:")
        print(f"  Shape: {w.shape}")
        print(f"  Input range: [{w.min():.4f}, {w.max():.4f}]")
        print(f"  Output range: [{w_q.min():.4f}, {w_q.max():.4f}]")
        print(f"  Has NaN: {has_nan}")
        print(f"  Has Inf: {has_inf}")
        print(f"  MSE: {mse:.6e}")
        print(f"  Max error: {max_error:.4f}")
        
        if has_nan:
            print("  ⚠️  WARNING: Output contains NaN!")
        elif has_inf:
            print("  ⚠️  WARNING: Output contains Inf!")
        else:
            print("  ✓ OK")
    
    print("\n--- Activation Quantization (E5M2) ---")
    activations = [
        ("Typical activation", torch.randn(32, 1024)),
        ("ReLU output", torch.relu(torch.randn(32, 1024))),
        ("Sparse activation", torch.randn(32, 1024) * (torch.rand(32, 1024) > 0.7)),
    ]
    
    for name, x in activations:
        x_q = quantizer.quantize_activation(x)
        
        # Check for NaN/Inf
        has_nan = torch.isnan(x_q).any().item()
        has_inf = torch.isinf(x_q).any().item()
        
        # Calculate error
        mse = ((x - x_q) ** 2).mean().item()
        
        print(f"\n{name}:")
        print(f"  Shape: {x.shape}")
        print(f"  Input range: [{x.min():.4f}, {x.max():.4f}]")
        print(f"  Output range: [{x_q.min():.4f}, {x_q.max():.4f}]")
        print(f"  Has NaN: {has_nan}")
        print(f"  Has Inf: {has_inf}")
        print(f"  MSE: {mse:.6e}")
        
        if has_nan:
            print("  ⚠️  WARNING: Output contains NaN!")
        elif has_inf:
            print("  ⚠️  WARNING: Output contains Inf!")
        else:
            print("  ✓ OK")
    
    print("\n" + "="*80)
    print("✓ All tests completed successfully!")
    print("="*80)

if __name__ == "__main__":
    main()

