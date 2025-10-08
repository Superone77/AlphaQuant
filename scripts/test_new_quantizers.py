#!/usr/bin/env python3
"""
Test script for all new quantizers (FP8, FP4, INT2, INT4, INT6, INT8).
This script verifies that all quantizers work correctly and can be used in the quantization pipeline.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from alphaquant.quantizers import (
    FP4Quantizer, FP4Config,
    FP8Quantizer, FP8Config,
    INT2Quantizer, INT2Config,
    INT4Quantizer, INT4Config,
    INT6Quantizer, INT6Config,
    INT8Quantizer, INT8Config,
    MinMaxObserver
)
from alphaquant.utils.replacement import create_quantizer_from_scheme


def test_quantizer_basic(quantizer_class, config_class, name, **config_kwargs):
    """Test basic functionality of a quantizer."""
    print(f"\n=== Testing {name} Quantizer ===")
    
    # Create quantizer
    config = config_class(**config_kwargs)
    quantizer = quantizer_class(config)
    
    # Test data
    x = torch.randn(2, 32, device='cuda' if torch.cuda.is_available() else 'cpu') * 10
    
    print(f"Original tensor shape: {x.shape}")
    print(f"Original tensor range: [{x.min():.3f}, {x.max():.3f}]")
    
    # Test weight quantization
    try:
        w_quant = quantizer.quantize_weight(x)
        print(f"Weight quantization successful. Shape: {w_quant.shape}")
        print(f"Quantized range: [{w_quant.min():.3f}, {w_quant.max():.3f}]")
    except Exception as e:
        print(f"Weight quantization failed: {e}")
        return False
    
    # Test activation quantization (dynamic)
    try:
        a_quant = quantizer.quantize_activation(x)
        print(f"Activation quantization (dynamic) successful. Shape: {a_quant.shape}")
        print(f"Quantized range: [{a_quant.min():.3f}, {a_quant.max():.3f}]")
    except Exception as e:
        print(f"Activation quantization (dynamic) failed: {e}")
        return False
    
    # Test activation quantization (static with observer)
    try:
        observer = MinMaxObserver()
        quantizer.attach_observer(observer)
        
        # Simulate calibration
        for _ in range(5):
            calib_data = torch.randn_like(x) * 5
            observer.observe(calib_data)
        
        quantizer.calibrate_finish()
        a_quant_static = quantizer.quantize_activation(x)
        print(f"Activation quantization (static) successful. Shape: {a_quant_static.shape}")
        print(f"Quantized range: [{a_quant_static.min():.3f}, {a_quant_static.max():.3f}]")
    except Exception as e:
        print(f"Activation quantization (static) failed: {e}")
        return False
    
    print(f"✅ {name} quantizer test passed!")
    return True


def test_quantizer_registry():
    """Test that quantizers can be created through the registry."""
    print(f"\n=== Testing Quantizer Registry ===")
    
    schemes = ['fp4', 'fp8', 'int2', 'int4', 'int6', 'int8']
    
    for scheme in schemes:
        try:
            (WQ, WCfg), (AQ, ACfg) = create_quantizer_from_scheme(
                {'wq': scheme, 'aq': scheme}, 'bfloat16'
            )
            print(f"✅ Registry lookup for {scheme} successful")
        except Exception as e:
            print(f"❌ Registry lookup for {scheme} failed: {e}")
            return False
    
    return True


def test_quantizer_integration():
    """Test quantizers with a simple linear layer."""
    print(f"\n=== Testing Quantizer Integration ===")
    
    # Create a simple linear layer
    linear = nn.Linear(32, 16, bias=True)
    if torch.cuda.is_available():
        linear = linear.cuda()
    
    # Test with different quantizers
    quantizers_to_test = [
        (INT8Quantizer, INT8Config, 'int8'),
        (INT4Quantizer, INT4Config, 'int4'),
        (FP8Quantizer, FP8Config, 'fp8'),
    ]
    
    for quantizer_class, config_class, name in quantizers_to_test:
        try:
            config = config_class()
            quantizer = quantizer_class(config)
            
            # Quantize weights
            w_quant = quantizer.quantize_weight(linear.weight)
            
            # Test forward pass
            x = torch.randn(2, 32, device=linear.weight.device)
            a_quant = quantizer.quantize_activation(x)
            
            # Simulate quantized forward pass
            output = torch.matmul(a_quant, w_quant.t())
            if linear.bias is not None:
                output = output + linear.bias
            
            print(f"✅ {name} integration test passed. Output shape: {output.shape}")
            
        except Exception as e:
            print(f"❌ {name} integration test failed: {e}")
            return False
    
    return True


def main():
    """Run all quantizer tests."""
    print("Starting quantizer tests...")
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Test individual quantizers
    quantizer_tests = [
        (FP4Quantizer, FP4Config, 'FP4', {'format': 'e2m1'}),
        (FP8Quantizer, FP8Config, 'FP8', {'format': 'e4m3'}),
        (INT2Quantizer, INT2Config, 'INT2', {'symmetric': True}),
        (INT4Quantizer, INT4Config, 'INT4', {'symmetric': True}),
        (INT6Quantizer, INT6Config, 'INT6', {'symmetric': True}),
        (INT8Quantizer, INT8Config, 'INT8', {'symmetric': True}),
    ]
    
    all_passed = True
    
    for quantizer_class, config_class, name, config_kwargs in quantizer_tests:
        if not test_quantizer_basic(quantizer_class, config_class, name, **config_kwargs):
            all_passed = False
    
    # Test registry
    if not test_quantizer_registry():
        all_passed = False
    
    # Test integration
    if not test_quantizer_integration():
        all_passed = False
    
    # Summary
    print(f"\n{'='*50}")
    if all_passed:
        print("🎉 All quantizer tests passed!")
    else:
        print("❌ Some quantizer tests failed!")
    print(f"{'='*50}")
    
    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
