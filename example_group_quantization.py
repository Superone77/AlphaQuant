#!/usr/bin/env python3
"""
Example script demonstrating group quantization for INT and FP quantizers.

Group quantization divides tensors into groups and computes quantization
parameters separately for each group, which can improve quantization accuracy.

Usage:
    python example_group_quantization.py
"""

import torch
from alphaquant.quantizers.int_quantizers import INT4Config, INT4Quantizer, INT8Config, INT8Quantizer
from alphaquant.quantizers.fp4 import FP4Config, FP4Quantizer
from alphaquant.quantizers.fp8 import FP8Config, FP8Quantizer


def compare_quantization(w, quantizer_no_group, quantizer_with_group, name):
    """Compare quantization with and without group quantization."""
    print(f"\n{'='*60}")
    print(f"{name} Quantization Comparison")
    print(f"{'='*60}")
    print(f"Original weight shape: {w.shape}")
    print(f"Original weight stats - mean: {w.mean():.6f}, std: {w.std():.6f}")
    print(f"Original weight range: [{w.min():.6f}, {w.max():.6f}]")
    
    # Quantize without group quantization
    w_quant_no_group = quantizer_no_group.quantize_weight(w)
    error_no_group = (w - w_quant_no_group).abs().mean()
    print(f"\nWithout group quantization:")
    print(f"  Mean absolute error: {error_no_group:.6f}")
    
    # Quantize with group quantization
    w_quant_with_group = quantizer_with_group.quantize_weight(w)
    error_with_group = (w - w_quant_with_group).abs().mean()
    print(f"\nWith group quantization (group_size={quantizer_with_group.group_size}):")
    print(f"  Mean absolute error: {error_with_group:.6f}")
    
    # Calculate improvement
    improvement = ((error_no_group - error_with_group) / error_no_group * 100)
    print(f"\nImprovement: {improvement:.2f}%")
    
    return error_no_group, error_with_group


def main():
    print("="*60)
    print("Group Quantization Demo")
    print("="*60)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Create a test weight tensor with varying magnitudes
    # This simulates real neural network weights with different scales
    w = torch.randn(512, 512) * 0.1
    # Add some regions with larger magnitudes
    w[100:200, 100:200] *= 5.0
    w[300:400, 300:400] *= 0.1
    
    print(f"\nTest tensor shape: {w.shape}")
    print(f"Test tensor stats - mean: {w.mean():.6f}, std: {w.std():.6f}")
    
    # ========================================================================
    # INT4 Quantization
    # ========================================================================
    print("\n" + "="*60)
    print("INT4 Quantization")
    print("="*60)
    
    # Without group quantization
    int4_cfg_no_group = INT4Config(
        use_group_quant=False,
        symmetric=True
    )
    int4_quantizer_no_group = INT4Quantizer(int4_cfg_no_group)
    
    # With group quantization (group_size=128)
    int4_cfg_with_group = INT4Config(
        use_group_quant=True,
        group_size=128,
        symmetric=True
    )
    int4_quantizer_with_group = INT4Quantizer(int4_cfg_with_group)
    
    compare_quantization(w, int4_quantizer_no_group, int4_quantizer_with_group, "INT4")
    
    # ========================================================================
    # INT8 Quantization
    # ========================================================================
    print("\n" + "="*60)
    print("INT8 Quantization")
    print("="*60)
    
    # Without group quantization
    int8_cfg_no_group = INT8Config(
        use_group_quant=False,
        symmetric=True
    )
    int8_quantizer_no_group = INT8Quantizer(int8_cfg_no_group)
    
    # With group quantization (group_size=256)
    int8_cfg_with_group = INT8Config(
        use_group_quant=True,
        group_size=256,
        symmetric=True
    )
    int8_quantizer_with_group = INT8Quantizer(int8_cfg_with_group)
    
    compare_quantization(w, int8_quantizer_no_group, int8_quantizer_with_group, "INT8")
    
    # ========================================================================
    # FP4 Quantization
    # ========================================================================
    print("\n" + "="*60)
    print("FP4 Quantization")
    print("="*60)
    
    # Without group quantization
    fp4_cfg_no_group = FP4Config(
        use_group_quant=False,
        format="e2m1"
    )
    fp4_quantizer_no_group = FP4Quantizer(fp4_cfg_no_group)
    
    # With group quantization (group_size=128)
    fp4_cfg_with_group = FP4Config(
        use_group_quant=True,
        group_size=128,
        format="e2m1"
    )
    fp4_quantizer_with_group = FP4Quantizer(fp4_cfg_with_group)
    
    compare_quantization(w, fp4_quantizer_no_group, fp4_quantizer_with_group, "FP4")
    
    # ========================================================================
    # FP8 Quantization
    # ========================================================================
    print("\n" + "="*60)
    print("FP8 Quantization")
    print("="*60)
    
    # Without group quantization
    fp8_cfg_no_group = FP8Config(
        use_group_quant=False,
        format="e4m3"
    )
    fp8_quantizer_no_group = FP8Quantizer(fp8_cfg_no_group)
    
    # With group quantization (group_size=128)
    fp8_cfg_with_group = FP8Config(
        use_group_quant=True,
        group_size=128,
        format="e4m3"
    )
    fp8_quantizer_with_group = FP8Quantizer(fp8_cfg_with_group)
    
    compare_quantization(w, fp8_quantizer_no_group, fp8_quantizer_with_group, "FP8")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print("\nGroup quantization parameters:")
    print("  - use_group_quant: Enable/disable group quantization (default: False)")
    print("  - group_size: Number of elements per group (default: 128)")
    print("\nKey benefits:")
    print("  - Better accuracy for weights with varying magnitudes")
    print("  - More granular quantization parameters")
    print("  - Flexible group_size allows trading off accuracy vs. overhead")
    print("\nUsage in config:")
    print('  INT4Config(use_group_quant=True, group_size=128)')
    print('  FP8Config(use_group_quant=True, group_size=256)')
    print("="*60)


if __name__ == "__main__":
    main()

