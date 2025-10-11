#!/usr/bin/env python3
"""Test script to verify MXFP8 NaN fix"""

import torch
from alphaquant.quantizers.kernel.fp_torch import mxfp8_torch

# Test cases that might produce NaN
test_cases = [
    ("Normal values", torch.randn(2, 32)),
    ("Very small values", torch.randn(2, 32) * 1e-10),
    ("All zeros", torch.zeros(2, 32)),
    ("Mix of zeros and values", torch.cat([torch.zeros(1, 32), torch.randn(1, 32)], dim=0)),
    ("Very large values", torch.randn(2, 32) * 1e10),
    ("Single zero row", torch.cat([torch.zeros(1, 32), torch.ones(1, 32)], dim=0)),
    ("Input with NaN", torch.tensor([[float('nan')] * 32, [1.0] * 32])),
    ("Input with Inf", torch.tensor([[float('inf')] * 32, [1.0] * 32])),
]

print("="*80)
print("Testing MXFP8 quantization for NaN issues (AFTER FIX)")
print("="*80)

all_passed = True

for name, x in test_cases:
    print(f"\nTest: {name}")
    print(f"  Input shape: {x.shape}")
    
    # Test E4M3
    try:
        x_q_e4m3 = mxfp8_torch(x.clone(), scaled_value_format="e4m3")
        has_nan_e4m3 = torch.isnan(x_q_e4m3).any().item()
        has_inf_e4m3 = torch.isinf(x_q_e4m3).any().item()
        
        if has_nan_e4m3:
            print(f"  ❌ E4M3 FAILED: Output has NaN!")
            all_passed = False
        elif has_inf_e4m3:
            print(f"  ⚠️  E4M3 WARNING: Output has Inf")
        else:
            print(f"  ✓ E4M3 PASSED: No NaN or Inf")
    except Exception as e:
        print(f"  ❌ E4M3 ERROR: {e}")
        all_passed = False
    
    # Test E5M2
    try:
        x_q_e5m2 = mxfp8_torch(x.clone(), scaled_value_format="e5m2")
        has_nan_e5m2 = torch.isnan(x_q_e5m2).any().item()
        has_inf_e5m2 = torch.isinf(x_q_e5m2).any().item()
        
        if has_nan_e5m2:
            print(f"  ❌ E5M2 FAILED: Output has NaN!")
            all_passed = False
        elif has_inf_e5m2:
            print(f"  ⚠️  E5M2 WARNING: Output has Inf")
        else:
            print(f"  ✓ E5M2 PASSED: No NaN or Inf")
    except Exception as e:
        print(f"  ❌ E5M2 ERROR: {e}")
        all_passed = False

print("\n" + "="*80)
if all_passed:
    print("✓ ALL TESTS PASSED - No NaN in outputs!")
else:
    print("❌ SOME TESTS FAILED - Please review")
print("="*80)

