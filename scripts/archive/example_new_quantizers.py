#!/usr/bin/env python3
"""
Example script demonstrating how to use the new quantizers (FP8, FP4, INT2, INT4, INT6, INT8).
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


def example_basic_usage():
    """Example of basic quantizer usage."""
    print("=== Basic Quantizer Usage ===")
    
    # Create test data
    x = torch.randn(2, 32) * 10
    print(f"Original tensor: {x[0, :8]}")
    
    # Example 1: INT8 quantization
    print("\n1. INT8 Quantization:")
    int8_config = INT8Config(symmetric=True)
    int8_quantizer = INT8Quantizer(int8_config)
    
    x_int8 = int8_quantizer.quantize_weight(x)
    print(f"INT8 quantized: {x_int8[0, :8]}")
    
    # Example 2: FP8 quantization
    print("\n2. FP8 Quantization:")
    fp8_config = FP8Config(format='e4m3')
    fp8_quantizer = FP8Quantizer(fp8_config)
    
    x_fp8 = fp8_quantizer.quantize_weight(x)
    print(f"FP8 quantized: {x_fp8[0, :8]}")
    
    # Example 3: INT4 quantization
    print("\n3. INT4 Quantization:")
    int4_config = INT4Config(symmetric=True)
    int4_quantizer = INT4Quantizer(int4_config)
    
    x_int4 = int4_quantizer.quantize_weight(x)
    print(f"INT4 quantized: {x_int4[0, :8]}")


def example_with_observer():
    """Example of using quantizers with observers for static quantization."""
    print("\n=== Using Observers for Static Quantization ===")
    
    # Create quantizer with observer
    config = INT8Config(symmetric=True)
    quantizer = INT8Quantizer(config)
    observer = MinMaxObserver()
    quantizer.attach_observer(observer)
    
    # Simulate calibration data
    print("Calibrating quantizer...")
    for i in range(10):
        calib_data = torch.randn(2, 32) * (i + 1)  # Different scales
        observer.observe(calib_data)
    
    # Finish calibration
    quantizer.calibrate_finish()
    
    # Test on new data
    test_data = torch.randn(2, 32) * 5
    quantized_data = quantizer.quantize_activation(test_data)
    
    print(f"Test data: {test_data[0, :8]}")
    print(f"Quantized: {quantized_data[0, :8]}")


def example_quantizer_registry():
    """Example of using quantizers through the registry system."""
    print("\n=== Using Quantizer Registry ===")
    
    # Create quantizers through registry
    schemes = ['int8', 'int4', 'fp8', 'fp4']
    
    for scheme in schemes:
        print(f"\nCreating {scheme} quantizer through registry:")
        (WQ, WCfg), (AQ, ACfg) = create_quantizer_from_scheme(
            {'wq': scheme, 'aq': scheme}, 'bfloat16'
        )
        
        # Create instances
        w_quantizer = WQ(WCfg())
        a_quantizer = AQ(ACfg())
        
        # Test quantization
        x = torch.randn(2, 32) * 10
        w_quant = w_quantizer.quantize_weight(x)
        a_quant = a_quantizer.quantize_activation(x)
        
        print(f"  Weight quantized range: [{w_quant.min():.3f}, {w_quant.max():.3f}]")
        print(f"  Activation quantized range: [{a_quant.min():.3f}, {a_quant.max():.3f}]")


def example_model_quantization():
    """Example of quantizing a simple model."""
    print("\n=== Model Quantization Example ===")
    
    # Create a simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(32, 16)
            self.linear2 = nn.Linear(16, 8)
        
        def forward(self, x):
            x = torch.relu(self.linear1(x))
            x = self.linear2(x)
            return x
    
    model = SimpleModel()
    print(f"Original model: {model}")
    
    # Create quantizers for each layer
    int8_config = INT8Config(symmetric=True)
    int8_quantizer = INT8Quantizer(int8_config)
    
    # Quantize weights
    model.linear1.weight.data = int8_quantizer.quantize_weight(model.linear1.weight.data)
    model.linear2.weight.data = int8_quantizer.quantize_weight(model.linear2.weight.data)
    
    print("Model weights quantized with INT8")
    
    # Test forward pass
    x = torch.randn(2, 32)
    with torch.no_grad():
        output = model(x)
    
    print(f"Model output shape: {output.shape}")
    print(f"Model output range: [{output.min():.3f}, {output.max():.3f}]")


def main():
    """Run all examples."""
    print("New Quantizers Examples")
    print("=" * 50)
    
    try:
        example_basic_usage()
        example_with_observer()
        example_quantizer_registry()
        example_model_quantization()
        
        print("\n" + "=" * 50)
        print("✅ All examples completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
