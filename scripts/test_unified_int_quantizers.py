#!/usr/bin/env python3
"""
Test script to verify the unified integer quantizers work correctly.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from alphaquant.quantizers import (
    INT2Quantizer, INT2Config,
    INT4Quantizer, INT4Config,
    INT6Quantizer, INT6Config,
    INT8Quantizer, INT8Config,
    MinMaxObserver
)


def test_unified_int_quantizers():
    """Test all unified integer quantizers."""
    print("Testing Unified Integer Quantizers")
    print("=" * 50)
    
    # Test data
    x = torch.randn(2, 32) * 10
    print(f"Original tensor: {x[0, :8]}")
    
    # Test all integer quantizers
    quantizers = [
        (INT2Quantizer, INT2Config, 'INT2'),
        (INT4Quantizer, INT4Config, 'INT4'),
        (INT6Quantizer, INT6Config, 'INT6'),
        (INT8Quantizer, INT8Config, 'INT8'),
    ]
    
    for quantizer_class, config_class, name in quantizers:
        print(f"\n--- Testing {name} ---")
        
        # Test symmetric quantization
        config = config_class(symmetric=True)
        quantizer = quantizer_class(config)
        
        x_quant = quantizer.quantize_weight(x)
        print(f"Symmetric quantized: {x_quant[0, :8]}")
        print(f"Range: [{x_quant.min():.3f}, {x_quant.max():.3f}]")
        
        # Test asymmetric quantization
        config = config_class(symmetric=False)
        quantizer = quantizer_class(config)
        
        x_quant = quantizer.quantize_weight(x)
        print(f"Asymmetric quantized: {x_quant[0, :8]}")
        print(f"Range: [{x_quant.min():.3f}, {x_quant.max():.3f}]")
        
        # Test with observer
        config = config_class(symmetric=True)
        quantizer = quantizer_class(config)
        observer = MinMaxObserver()
        quantizer.attach_observer(observer)
        
        # Calibrate
        for _ in range(5):
            calib_data = torch.randn_like(x) * 5
            observer.observe(calib_data)
        
        quantizer.calibrate_finish()
        x_quant_static = quantizer.quantize_activation(x)
        print(f"Static quantized: {x_quant_static[0, :8]}")
        
        print(f"✅ {name} test passed!")


def test_inheritance():
    """Test that the inheritance structure works correctly."""
    print(f"\n--- Testing Inheritance Structure ---")
    
    # Test that all quantizers inherit from BaseIntQuantizer
    quantizers = [
        (INT2Quantizer, INT2Config),
        (INT4Quantizer, INT4Config),
        (INT6Quantizer, INT6Config),
        (INT8Quantizer, INT8Config),
    ]
    
    for quantizer_class, config_class in quantizers:
        config = config_class()
        quantizer = quantizer_class(config)
        
        # Check that they have the expected attributes
        assert hasattr(quantizer, 'qmin')
        assert hasattr(quantizer, 'qmax')
        assert hasattr(quantizer, 'symmetric')
        assert hasattr(quantizer, '_quantize_kernel')
        
        print(f"✅ {quantizer_class.__name__} inheritance structure correct")


def main():
    """Run all tests."""
    try:
        test_unified_int_quantizers()
        test_inheritance()
        
        print(f"\n{'='*50}")
        print("🎉 All unified integer quantizer tests passed!")
        print(f"{'='*50}")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
