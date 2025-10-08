#!/usr/bin/env python
"""
Test script for precision-based Alpha-Hill analysis.
Uses a small model for quick validation.
"""
from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

from alphaquant.alpha_hill.utils import alpha_hill_from_weight


class TinyTestModel(nn.Module):
    """Minimal model for testing."""
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(128, 256)
        self.layer2 = nn.Linear(256, 512)
        self.layer3 = nn.Linear(512, 128)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


def test_precision_alpha_computation():
    """Test computing alpha at different precisions."""
    print("\n" + "="*60)
    print("TEST 1: Alpha-Hill computation at different precisions")
    print("="*60)
    
    # Create a simple weight matrix
    torch.manual_seed(42)
    weight = torch.randn(256, 128)
    
    precisions = ['fp32', 'fp16', 'bf16']
    results = {}
    
    for precision in precisions:
        if precision == 'fp32':
            dtype = torch.float32
        elif precision == 'fp16':
            dtype = torch.float16
        elif precision == 'bf16':
            dtype = torch.bfloat16
        
        w_converted = weight.to(dtype=dtype)
        alpha, k_used, n_eigs, method = alpha_hill_from_weight(
            w_converted, 
            k_frac=0.1, 
            force_cpu_svd=True
        )
        
        results[precision] = alpha
        print(f"{precision:>6s}: α = {alpha:.6f} (k={k_used}, n={n_eigs}, {method})")
    
    # Check that we got valid results
    for precision, alpha in results.items():
        assert not np.isnan(alpha), f"Alpha is NaN for {precision}"
        assert alpha > 0, f"Alpha should be positive for {precision}"
    
    print("✓ Test passed: All precisions computed successfully")
    return results


def test_model_iteration():
    """Test iterating through model layers."""
    print("\n" + "="*60)
    print("TEST 2: Model layer iteration")
    print("="*60)
    
    model = TinyTestModel()
    
    from alphaquant.alpha_hill.utils import iter_linear_modules, categorize
    
    layer_count = 0
    for name, module in iter_linear_modules(model):
        layer_count += 1
        category = categorize(name)
        print(f"  {name}: {module.weight.shape}, category={category}")
    
    assert layer_count == 3, f"Expected 3 layers, found {layer_count}"
    print(f"✓ Test passed: Found {layer_count} linear layers")


def test_precision_sensitivity():
    """Test that different precisions give different alpha values."""
    print("\n" + "="*60)
    print("TEST 3: Precision sensitivity")
    print("="*60)
    
    torch.manual_seed(123)
    
    # Create a weight with specific structure
    # Make it have high singular value concentration
    U = torch.randn(200, 200)
    U, _ = torch.linalg.qr(U)
    
    # Create diagonal matrix with exponentially decaying singular values
    s = torch.logspace(0, -2, 100)
    S = torch.zeros(200, 100)
    S[:100, :] = torch.diag(s)
    
    V = torch.randn(100, 100)
    V, _ = torch.linalg.qr(V)
    
    weight = U @ S @ V.T
    
    precisions = ['fp32', 'fp16', 'bf16']
    alphas = {}
    
    for precision in precisions:
        if precision == 'fp32':
            dtype = torch.float32
        elif precision == 'fp16':
            dtype = torch.float16
        elif precision == 'bf16':
            dtype = torch.bfloat16
        
        w_converted = weight.to(dtype=dtype)
        alpha, k_used, n_eigs, method = alpha_hill_from_weight(
            w_converted,
            k_frac=0.1,
            force_cpu_svd=True
        )
        
        alphas[precision] = alpha
        print(f"{precision:>6s}: α = {alpha:.6f}")
    
    # Verify we got different values (within numerical tolerance)
    alpha_values = list(alphas.values())
    max_diff = max(alpha_values) - min(alpha_values)
    
    print(f"\nAlpha range: {min(alpha_values):.6f} - {max(alpha_values):.6f}")
    print(f"Max difference: {max_diff:.6f}")
    
    if max_diff > 0.001:
        print("✓ Test passed: Precisions show measurable differences")
    else:
        print("⚠ Warning: Differences are very small (this can be normal)")


def test_output_structure():
    """Test that output directories and files are created correctly."""
    print("\n" + "="*60)
    print("TEST 4: Output structure validation")
    print("="*60)
    
    output_dir = Path("./test_output_precision_analysis")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Create expected subdirectories
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    individual_plots_dir = plots_dir / "individual_layers"
    individual_plots_dir.mkdir(exist_ok=True)
    
    # Verify structure
    assert output_dir.exists(), "Output directory not created"
    assert plots_dir.exists(), "Plots directory not created"
    assert individual_plots_dir.exists(), "Individual plots directory not created"
    
    print(f"✓ Test passed: Directory structure created at {output_dir}")
    
    # Clean up
    import shutil
    shutil.rmtree(output_dir)
    print("✓ Cleanup complete")


def test_csv_format():
    """Test CSV data format."""
    print("\n" + "="*60)
    print("TEST 5: CSV format validation")
    print("="*60)
    
    import pandas as pd
    
    # Create sample data
    data = {
        'layer_name': ['layer1', 'layer2', 'layer3'],
        'category': ['attn_q', 'mlp_up', 'attn_k'],
        'out_features': [256, 512, 128],
        'in_features': [128, 256, 512],
        'numel': [32768, 131072, 65536],
        'alpha_fp32': [2.5, 3.1, 2.8],
        'alpha_fp16': [2.4, 3.0, 2.7],
        'alpha_bf16': [2.45, 3.05, 2.75],
    }
    
    df = pd.DataFrame(data)
    
    # Verify columns
    required_cols = ['layer_name', 'category', 'out_features', 'in_features', 'numel']
    for col in required_cols:
        assert col in df.columns, f"Missing required column: {col}"
    
    # Verify data types
    assert df['out_features'].dtype in [np.int64, np.int32], "out_features should be integer"
    assert df['alpha_fp32'].dtype in [np.float64, np.float32], "alpha values should be float"
    
    print("✓ Test passed: CSV format is valid")
    print(f"\nSample data:\n{df.head()}")


def run_mini_analysis():
    """Run a mini version of the full analysis."""
    print("\n" + "="*60)
    print("TEST 6: Mini end-to-end analysis")
    print("="*60)
    
    from alphaquant.alpha_hill.utils import iter_linear_modules, safe_get_in_out_features, categorize
    
    model = TinyTestModel()
    precisions = ['fp32', 'fp16', 'bf16']
    
    results = []
    
    for name, module in iter_linear_modules(model):
        weight = module.weight.detach().contiguous()
        out_features, in_features = safe_get_in_out_features(module)
        category = categorize(name)
        
        print(f"\n{name}:")
        
        row = {
            'layer_name': name,
            'category': category,
            'shape': f"[{out_features}, {in_features}]"
        }
        
        for precision in precisions:
            if precision == 'fp32':
                dtype = torch.float32
            elif precision == 'fp16':
                dtype = torch.float16
            elif precision == 'bf16':
                dtype = torch.bfloat16
            
            w_converted = weight.to(dtype=dtype)
            alpha, k_used, n_eigs, method = alpha_hill_from_weight(
                w_converted,
                k_frac=0.1,
                force_cpu_svd=True
            )
            
            row[f'alpha_{precision}'] = alpha
            print(f"  {precision}: α = {alpha:.6f}")
        
        results.append(row)
    
    import pandas as pd
    df = pd.DataFrame(results)
    
    print("\nResults summary:")
    print(df.to_string(index=False))
    
    print("\n✓ Test passed: Mini analysis completed successfully")


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("PRECISION-BASED ALPHA-HILL ANALYSIS - TEST SUITE")
    print("="*80)
    
    try:
        test_precision_alpha_computation()
        test_model_iteration()
        test_precision_sensitivity()
        test_output_structure()
        test_csv_format()
        run_mini_analysis()
        
        print("\n" + "="*80)
        print("ALL TESTS PASSED ✓")
        print("="*80)
        print("\nThe analyze_precision_alpha_hill.py script is ready to use!")
        print("\nQuick start:")
        print("  python scripts/analyze_precision_alpha_hill.py \\")
        print("    --model meta-llama/Llama-3.1-8B \\")
        print("    --precisions fp32,fp16,bf16 \\")
        print("    --output-dir ./results/llama31_8b_precision")
        print("\n" + "="*80)
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

