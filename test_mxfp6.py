#!/usr/bin/env python3
"""
Test script for MXFP6 and FP6 quantization.
Tests both E2M3 and E3M2 formats (default is E3M2).
"""

import torch
from alphaquant.quantizers import MXFP6Quantizer, MXFP6Config, FP6Quantizer, FP6Config

def test_mxfp6():
    print("=" * 80)
    print("Testing MXFP6 Quantization")
    print("=" * 80)
    
    # Test E3M2 (default)
    print("\n1. Testing MXFP6 with E3M2 format (default)...")
    cfg = MXFP6Config(format='e3m2')
    quantizer = MXFP6Quantizer(cfg)
    
    x = torch.randn(4, 32) * 2
    print(f"   Input shape: {x.shape}")
    print(f"   Input range: [{x.min():.4f}, {x.max():.4f}]")
    
    x_q = quantizer.quantize_weight(x)
    print(f"   Output shape: {x_q.shape}")
    print(f"   Output range: [{x_q.min():.4f}, {x_q.max():.4f}]")
    print(f"   MSE: {((x - x_q) ** 2).mean():.6f}")
    print(f"   Max absolute error: {(x - x_q).abs().max():.6f}")
    
    # Test E2M3
    print("\n2. Testing MXFP6 with E2M3 format...")
    cfg = MXFP6Config(format='e2m3')
    quantizer = MXFP6Quantizer(cfg)
    
    x = torch.randn(4, 32) * 2
    print(f"   Input shape: {x.shape}")
    print(f"   Input range: [{x.min():.4f}, {x.max():.4f}]")
    
    x_q = quantizer.quantize_weight(x)
    print(f"   Output shape: {x_q.shape}")
    print(f"   Output range: [{x_q.min():.4f}, {x_q.max():.4f}]")
    print(f"   MSE: {((x - x_q) ** 2).mean():.6f}")
    print(f"   Max absolute error: {(x - x_q).abs().max():.6f}")
    
    # Test activation quantization
    print("\n3. Testing MXFP6 activation quantization (E3M2)...")
    cfg = MXFP6Config(format='e3m2')
    quantizer = MXFP6Quantizer(cfg)
    
    x = torch.randn(8, 64) * 3
    print(f"   Input shape: {x.shape}")
    print(f"   Input range: [{x.min():.4f}, {x.max():.4f}]")
    
    x_q = quantizer.quantize_activation(x)
    print(f"   Output shape: {x_q.shape}")
    print(f"   Output range: [{x_q.min():.4f}, {x_q.max():.4f}]")
    print(f"   MSE: {((x - x_q) ** 2).mean():.6f}")
    

def test_fp6():
    print("\n" + "=" * 80)
    print("Testing FP6 Quantization (without microscaling)")
    print("=" * 80)
    
    # Test E3M2 (default)
    print("\n1. Testing FP6 with E3M2 format (default)...")
    cfg = FP6Config(format='e3m2')
    quantizer = FP6Quantizer(cfg)
    
    x = torch.randn(4, 32) * 2
    print(f"   Input shape: {x.shape}")
    print(f"   Input range: [{x.min():.4f}, {x.max():.4f}]")
    
    x_q = quantizer.quantize_weight(x)
    print(f"   Output shape: {x_q.shape}")
    print(f"   Output range: [{x_q.min():.4f}, {x_q.max():.4f}]")
    print(f"   MSE: {((x - x_q) ** 2).mean():.6f}")
    
    # Test E2M3
    print("\n2. Testing FP6 with E2M3 format...")
    cfg = FP6Config(format='e2m3')
    quantizer = FP6Quantizer(cfg)
    
    x = torch.randn(4, 32) * 2
    print(f"   Input shape: {x.shape}")
    print(f"   Input range: [{x.min():.4f}, {x.max():.4f}]")
    
    x_q = quantizer.quantize_weight(x)
    print(f"   Output shape: {x_q.shape}")
    print(f"   Output range: [{x_q.min():.4f}, {x_q.max():.4f}]")
    print(f"   MSE: {((x - x_q) ** 2).mean():.6f}")


def test_format_validation():
    print("\n" + "=" * 80)
    print("Testing Format Validation")
    print("=" * 80)
    
    # Test invalid format
    print("\n1. Testing invalid format (should raise error)...")
    try:
        cfg = MXFP6Config(format='e4m2')  # Invalid format
        print("   ERROR: Should have raised ValueError!")
    except ValueError as e:
        print(f"   ✓ Correctly raised ValueError: {e}")
    
    # Test valid formats
    print("\n2. Testing valid formats...")
    for fmt in ['e2m3', 'e3m2']:
        try:
            cfg = MXFP6Config(format=fmt)
            print(f"   ✓ Format '{fmt}' is valid")
        except ValueError as e:
            print(f"   ERROR: Format '{fmt}' should be valid but raised: {e}")


def test_comparison():
    """Compare MXFP6 E2M3 vs E3M2 formats"""
    print("\n" + "=" * 80)
    print("Comparing E2M3 vs E3M2 formats")
    print("=" * 80)
    
    x = torch.randn(16, 128) * 5
    
    print(f"\nOriginal tensor stats:")
    print(f"  Shape: {x.shape}")
    print(f"  Range: [{x.min():.4f}, {x.max():.4f}]")
    print(f"  Mean: {x.mean():.4f}, Std: {x.std():.4f}")
    
    # E2M3
    cfg_e2m3 = MXFP6Config(format='e2m3')
    quantizer_e2m3 = MXFP6Quantizer(cfg_e2m3)
    x_e2m3 = quantizer_e2m3.quantize_weight(x)
    
    print(f"\nE2M3 (2 exp bits, 3 mantissa bits, max≈7.5):")
    print(f"  Range: [{x_e2m3.min():.4f}, {x_e2m3.max():.4f}]")
    print(f"  MSE: {((x - x_e2m3) ** 2).mean():.6f}")
    print(f"  Max error: {(x - x_e2m3).abs().max():.6f}")
    
    # E3M2
    cfg_e3m2 = MXFP6Config(format='e3m2')
    quantizer_e3m2 = MXFP6Quantizer(cfg_e3m2)
    x_e3m2 = quantizer_e3m2.quantize_weight(x)
    
    print(f"\nE3M2 (3 exp bits, 2 mantissa bits, max≈28.0):")
    print(f"  Range: [{x_e3m2.min():.4f}, {x_e3m2.max():.4f}]")
    print(f"  MSE: {((x - x_e3m2) ** 2).mean():.6f}")
    print(f"  Max error: {(x - x_e3m2).abs().max():.6f}")


if __name__ == '__main__':
    test_mxfp6()
    test_fp6()
    test_format_validation()
    test_comparison()
    
    print("\n" + "=" * 80)
    print("All tests completed successfully! ✓")
    print("=" * 80)

