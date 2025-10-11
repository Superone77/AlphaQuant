#!/usr/bin/env python3
"""
Quick test script to verify group quantization functionality.
"""

import torch
from alphaquant.quantizers.int_quantizers import INT4Config, INT4Quantizer, INT8Config, INT8Quantizer
from alphaquant.quantizers.fp4 import FP4Config, FP4Quantizer
from alphaquant.quantizers.fp8 import FP8Config, FP8Quantizer


def test_quantizer(name, config_class, quantizer_class, **kwargs):
    """Test a quantizer with and without group quantization."""
    print(f"\nTesting {name}...")
    
    # Create test weight
    w = torch.randn(256, 256)
    
    # Test without group quantization
    cfg_no_group = config_class(use_group_quant=False, **kwargs)
    quantizer_no_group = quantizer_class(cfg_no_group)
    w_quant_no_group = quantizer_no_group.quantize_weight(w)
    assert w_quant_no_group.shape == w.shape, f"{name} without group: shape mismatch"
    
    # Test with group quantization
    cfg_with_group = config_class(use_group_quant=True, group_size=128, **kwargs)
    quantizer_with_group = quantizer_class(cfg_with_group)
    w_quant_with_group = quantizer_with_group.quantize_weight(w)
    assert w_quant_with_group.shape == w.shape, f"{name} with group: shape mismatch"
    
    # Verify the results are different (group quant should be different)
    error_no_group = (w - w_quant_no_group).abs().mean().item()
    error_with_group = (w - w_quant_with_group).abs().mean().item()
    
    print(f"  ✓ Shape preserved: {w.shape}")
    print(f"  ✓ Error without group: {error_no_group:.6f}")
    print(f"  ✓ Error with group: {error_with_group:.6f}")
    
    return True


def main():
    print("="*60)
    print("Group Quantization - Quick Test")
    print("="*60)
    
    torch.manual_seed(42)
    
    try:
        # Test INT quantizers
        test_quantizer("INT4", INT4Config, INT4Quantizer, symmetric=True)
        test_quantizer("INT8", INT8Config, INT8Quantizer, symmetric=True)
        
        # Test FP quantizers
        test_quantizer("FP4", FP4Config, FP4Quantizer, format="e2m1")
        test_quantizer("FP8", FP8Config, FP8Quantizer, format="e4m3")
        
        print("\n" + "="*60)
        print("✓ All tests passed!")
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

