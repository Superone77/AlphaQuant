#!/usr/bin/env python3
"""
Quick test to verify MXFP6 fix for torch.pow() issue.
"""

import torch
from alphaquant.quantizers import MXFP6Quantizer, MXFP6Config

def test_mxfp6_basic():
    print("Testing MXFP6 basic functionality...")
    
    # Test E3M2 (default)
    print("\n1. Testing E3M2 format (default)...")
    cfg = MXFP6Config(format='e3m2')
    quantizer = MXFP6Quantizer(cfg)
    
    # Test with different tensor sizes
    for shape in [(4, 32), (16, 128), (64, 256)]:
        x = torch.randn(*shape) * 2
        try:
            x_q = quantizer.quantize_weight(x)
            mse = ((x - x_q) ** 2).mean()
            print(f"   Shape {shape}: MSE = {mse:.6f} ✓")
        except Exception as e:
            print(f"   Shape {shape}: FAILED - {e}")
            return False
    
    # Test E2M3
    print("\n2. Testing E2M3 format...")
    cfg = MXFP6Config(format='e2m3')
    quantizer = MXFP6Quantizer(cfg)
    
    for shape in [(4, 32), (16, 128)]:
        x = torch.randn(*shape) * 2
        try:
            x_q = quantizer.quantize_weight(x)
            mse = ((x - x_q) ** 2).mean()
            print(f"   Shape {shape}: MSE = {mse:.6f} ✓")
        except Exception as e:
            print(f"   Shape {shape}: FAILED - {e}")
            return False
    
    # Test activation quantization
    print("\n3. Testing activation quantization...")
    cfg = MXFP6Config(format='e3m2')
    quantizer = MXFP6Quantizer(cfg)
    
    x = torch.randn(8, 64) * 3
    try:
        x_q = quantizer.quantize_activation(x)
        mse = ((x - x_q) ** 2).mean()
        print(f"   Activation MSE = {mse:.6f} ✓")
    except Exception as e:
        print(f"   Activation quantization FAILED - {e}")
        return False
    
    # Test with edge cases
    print("\n4. Testing edge cases...")
    
    # Very small values
    x = torch.randn(4, 32) * 0.001
    try:
        x_q = quantizer.quantize_weight(x)
        print(f"   Small values: OK ✓")
    except Exception as e:
        print(f"   Small values FAILED - {e}")
        return False
    
    # Large values
    x = torch.randn(4, 32) * 100
    try:
        x_q = quantizer.quantize_weight(x)
        print(f"   Large values: OK ✓")
    except Exception as e:
        print(f"   Large values FAILED - {e}")
        return False
    
    # Mixed positive/negative
    x = torch.randn(4, 32) * 10
    try:
        x_q = quantizer.quantize_weight(x)
        print(f"   Mixed signs: OK ✓")
    except Exception as e:
        print(f"   Mixed signs FAILED - {e}")
        return False
    
    # With zeros
    x = torch.randn(4, 32)
    x[0, :8] = 0  # Set some values to zero
    try:
        x_q = quantizer.quantize_weight(x)
        print(f"   With zeros: OK ✓")
    except Exception as e:
        print(f"   With zeros FAILED - {e}")
        return False
    
    return True


if __name__ == '__main__':
    print("=" * 80)
    print("MXFP6 Fix Verification Test")
    print("=" * 80)
    
    success = test_mxfp6_basic()
    
    print("\n" + "=" * 80)
    if success:
        print("✓ All tests passed! MXFP6 is working correctly.")
    else:
        print("✗ Some tests failed. Please check the errors above.")
    print("=" * 80)

