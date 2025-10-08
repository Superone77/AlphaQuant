#!/usr/bin/env python
"""
Analyze Alpha-Hill values across different quantization formats for each layer.
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alphaquant.alpha_hill.utils import (
    alpha_hill_from_weight,
    categorize,
    iter_linear_modules,
    safe_get_in_out_features,
    setup_logging
)
from alphaquant.utils.hf_utils import load_hf_causal_lm
from alphaquant.quantizers.mxfp4 import MXFP4Quantizer, MXFP4Config
from alphaquant.quantizers.mxfp8 import MXFP8Quantizer, MXFP8Config
from alphaquant.quantizers.fp4 import FP4Quantizer, FP4Config
from alphaquant.quantizers.fp8 import FP8Quantizer, FP8Config
from alphaquant.quantizers.int_quantizers import (
    INT2Quantizer, INT2Config,
    INT3Quantizer, INT3Config,
    INT4Quantizer, INT4Config,
    INT6Quantizer, INT6Config,
    INT8Quantizer, INT8Config
)


def apply_quantization(weight: torch.Tensor, quant_format: str) -> torch.Tensor:
    """Apply quantization to weight and return quantized weight."""
    quantizer_map = {
        'bf16': None,  # No quantization
        'mxfp8': (MXFP8Quantizer, MXFP8Config),
        'mxfp4': (MXFP4Quantizer, MXFP4Config),
        'fp8': (FP8Quantizer, FP8Config),
        'fp4': (FP4Quantizer, FP4Config),
        'int8': (INT8Quantizer, INT8Config),
        'int6': (INT6Quantizer, INT6Config),
        'int4': (INT4Quantizer, INT4Config),
        'int3': (INT3Quantizer, INT3Config),
        'int2': (INT2Quantizer, INT2Config),
    }
    
    if quant_format == 'bf16':
        # No quantization, just convert to bf16
        return weight.to(torch.bfloat16).to(torch.float32)
    
    if quant_format not in quantizer_map:
        raise ValueError(f"Unknown quantization format: {quant_format}")
    
    QuantizerClass, ConfigClass = quantizer_map[quant_format]
    config = ConfigClass(group_size=32, dtype='float32')
    quantizer = QuantizerClass(config)
    
    # Apply fake quantization (quant -> dequant)
    quantized_weight = quantizer.quantize_weight(weight)
    
    return quantized_weight


def compute_alpha_for_all_formats(
    weight: torch.Tensor,
    quant_formats: List[str],
    k_frac: float = 0.1
) -> Dict[str, float]:
    """Compute Alpha-Hill for all quantization formats."""
    results = {}
    
    for fmt in quant_formats:
        try:
            # Apply quantization
            quantized_weight = apply_quantization(weight, fmt)
            
            # Compute Alpha-Hill
            alpha, k_used, n_eigs, method = alpha_hill_from_weight(
                quantized_weight,
                k_frac=k_frac,
                force_cpu_svd=True,
                use_lowrank=False
            )
            
            results[fmt] = {
                'alpha': alpha,
                'k_used': k_used,
                'n_eigs': n_eigs,
                'method': method
            }
            
        except Exception as e:
            print(f"  Error with {fmt}: {e}")
            results[fmt] = {
                'alpha': float('nan'),
                'k_used': -1,
                'n_eigs': -1,
                'method': 'failed'
            }
    
    return results


def plot_layer_bar_chart(
    layer_name: str,
    quant_results: Dict[str, Dict],
    output_path: str,
    quant_formats: List[str]
):
    """Create bar chart for a single layer showing alpha across quantization formats."""
    alphas = [quant_results[fmt]['alpha'] for fmt in quant_formats]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(quant_formats)))
    bars = ax.bar(quant_formats, alphas, color=colors, alpha=0.8, edgecolor='black')
    
    # Add value labels on bars
    for bar, alpha in zip(bars, alphas):
        height = bar.get_height()
        if not np.isnan(height):
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                height,
                f'{alpha:.3f}',
                ha='center',
                va='bottom',
                fontsize=9,
                fontweight='bold'
            )
    
    ax.set_xlabel('Quantization Format', fontsize=12, fontweight='bold')
    ax.set_ylabel('Alpha-Hill Value', fontsize=12, fontweight='bold')
    ax.set_title(f'Alpha-Hill: {layer_name}', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Alpha-Hill across quantization formats"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HF model id or local path"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to load model (cpu or cuda)"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp32",
        help="Initial dtype to load model (fp32|fp16|bf16)"
    )
    parser.add_argument(
        "--quant-formats",
        type=str,
        default="bf16,mxfp8,mxfp4,fp8,fp4,int8,int6,int4",
        help="Comma-separated list of quantization formats"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results/quantization_alpha_analysis",
        help="Output directory"
    )
    parser.add_argument(
        "--k-frac",
        type=float,
        default=0.1,
        help="Fraction of eigenvalues for Alpha-Hill"
    )
    parser.add_argument(
        "--filter-layers",
        type=str,
        default=None,
        help="Regex pattern to filter layers"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup
    setup_logging(args.log_level)
    quant_formats = [f.strip() for f in args.quant_formats.split(',')]
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*80)
    print("QUANTIZATION-BASED ALPHA-HILL ANALYSIS")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Quantization formats: {quant_formats}")
    print(f"Output: {output_dir}")
    print("="*80 + "\n")
    
    # Load model
    print(f"Loading model...")
    model = load_hf_causal_lm(args.model, device=args.device, dtype=args.dtype)
    model.eval()
    
    # Analyze layers
    all_results = []
    layer_count = 0
    
    print("\nAnalyzing layers...\n")
    
    for name, module in iter_linear_modules(model):
        # Filter if needed
        if args.filter_layers:
            import re
            if not re.search(args.filter_layers, name):
                continue
        
        if not hasattr(module, 'weight') or module.weight is None:
            continue
        
        layer_count += 1
        weight = module.weight.detach().contiguous()
        out_features, in_features = safe_get_in_out_features(module)
        category = categorize(name)
        
        print(f"[{layer_count}] {name}")
        print(f"    Shape: [{out_features}, {in_features}], Category: {category}")
        
        # Compute alpha for all quantization formats
        quant_results = compute_alpha_for_all_formats(weight, quant_formats, args.k_frac)
        
        # Print results
        for fmt in quant_formats:
            alpha = quant_results[fmt]['alpha']
            if not np.isnan(alpha):
                print(f"    {fmt:>6s}: α = {alpha:.4f}")
            else:
                print(f"    {fmt:>6s}: FAILED")
        
        # Prepare CSV row
        result_row = {
            'layer_name': name,
            'category': category,
            'out_features': out_features,
            'in_features': in_features,
            'numel': int(weight.numel())
        }
        
        for fmt in quant_formats:
            result_row[f'alpha_{fmt}'] = quant_results[fmt]['alpha']
            result_row[f'k_used_{fmt}'] = quant_results[fmt]['k_used']
            result_row[f'n_eigs_{fmt}'] = quant_results[fmt]['n_eigs']
            result_row[f'method_{fmt}'] = quant_results[fmt]['method']
        
        all_results.append(result_row)
        
        # Plot
        plot_name = f"layer_{layer_count:03d}_{name.replace('.', '_')}.png"
        plot_path = plots_dir / plot_name
        plot_layer_bar_chart(name, quant_results, str(plot_path), quant_formats)
        
        print()
    
    print("="*80)
    print(f"Analyzed {layer_count} layers")
    print("="*80 + "\n")
    
    # Save CSV
    df = pd.DataFrame(all_results)
    csv_path = output_dir / "quantization_alpha_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"✓ Results saved to: {csv_path}")
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80 + "\n")
    
    for fmt in quant_formats:
        col = f'alpha_{fmt}'
        if col in df.columns:
            values = df[col].dropna()
            if len(values) > 0:
                print(f"{fmt:>6s}: mean={values.mean():.4f}, std={values.std():.4f}, "
                      f"min={values.min():.4f}, max={values.max():.4f}")
    
    # Summary by category
    print("\n" + "="*80)
    print("BY CATEGORY")
    print("="*80 + "\n")
    
    for category in df['category'].unique():
        cat_df = df[df['category'] == category]
        print(f"\n{category} ({len(cat_df)} layers):")
        for fmt in quant_formats:
            col = f'alpha_{fmt}'
            if col in cat_df.columns:
                values = cat_df[col].dropna()
                if len(values) > 0:
                    print(f"  {fmt:>6s}: {values.mean():.4f} ± {values.std():.4f}")
    
    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)
    print(f"\nOutput directory: {output_dir}")
    print(f"  - CSV: {csv_path.name}")
    print(f"  - Plots: {len(list(plots_dir.glob('*.png')))} files in plots/")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

