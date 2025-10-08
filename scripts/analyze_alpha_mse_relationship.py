#!/usr/bin/env python
"""
Analyze the relationship between Alpha-Hill values and quantization MSE.
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

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


def compute_mse(original: torch.Tensor, quantized: torch.Tensor) -> float:
    """Compute Mean Squared Error between original and quantized weights."""
    mse = torch.mean((original - quantized) ** 2).item()
    return mse


def apply_quantization_and_compute_mse(
    weight: torch.Tensor, 
    quant_format: str
) -> Tuple[torch.Tensor, float]:
    """Apply quantization to weight and compute MSE."""
    quantizer_map = {
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
    
    if quant_format not in quantizer_map:
        raise ValueError(f"Unknown quantization format: {quant_format}")
    
    QuantizerClass, ConfigClass = quantizer_map[quant_format]
    config = ConfigClass(group_size=32, dtype='float32')
    quantizer = QuantizerClass(config)
    
    # Apply fake quantization (quant -> dequant)
    quantized_weight = quantizer.quantize_weight(weight)
    
    # Compute MSE
    mse = compute_mse(weight, quantized_weight)
    
    return quantized_weight, mse


def compute_alpha_and_mse_for_all_formats(
    weight: torch.Tensor,
    quant_formats: List[str],
    k_frac: float = 0.1
) -> Tuple[float, Dict[str, float]]:
    """
    Compute Alpha-Hill for original weight and MSE for all quantization formats.
    
    Returns:
        alpha_original: Alpha-Hill value of the original weight
        mse_results: Dictionary mapping format to MSE
    """
    # Compute Alpha-Hill for original weight
    try:
        alpha_original, k_used, n_eigs, method = alpha_hill_from_weight(
            weight,
            k_frac=k_frac,
            force_cpu_svd=True,
            use_lowrank=False
        )
    except Exception as e:
        print(f"  Error computing alpha: {e}")
        alpha_original = float('nan')
    
    # Compute MSE for each quantization format
    mse_results = {}
    for fmt in quant_formats:
        try:
            _, mse = apply_quantization_and_compute_mse(weight, fmt)
            mse_results[fmt] = mse
        except Exception as e:
            print(f"  Error with {fmt}: {e}")
            mse_results[fmt] = float('nan')
    
    return alpha_original, mse_results


def plot_alpha_mse_scatter(
    df: pd.DataFrame,
    quant_formats: List[str],
    output_path: str
):
    """
    Create scatter plot showing relationship between Alpha-Hill and MSE.
    
    Each point represents a layer, colored by quantization format.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Color map for different quantization formats
    colors = plt.cm.tab10(np.linspace(0, 1, len(quant_formats)))
    color_map = dict(zip(quant_formats, colors))
    
    # Plot each quantization format
    for fmt in quant_formats:
        # Filter data for this format
        mask = df['quantization_format'] == fmt
        fmt_data = df[mask]
        
        # Remove NaN values
        fmt_data = fmt_data.dropna(subset=['alpha', 'mse'])
        
        if len(fmt_data) > 0:
            ax.scatter(
                fmt_data['alpha'],
                fmt_data['mse'],
                color=color_map[fmt],
                label=fmt,
                alpha=0.6,
                s=50,
                edgecolors='black',
                linewidth=0.5
            )
    
    ax.set_xlabel('Alpha-Hill Value', fontsize=14, fontweight='bold')
    ax.set_ylabel('MSE (Mean Squared Error)', fontsize=14, fontweight='bold')
    ax.set_title('Relationship between Alpha-Hill and Quantization MSE', 
                 fontsize=15, fontweight='bold')
    ax.set_yscale('log')  # Log scale for MSE (usually spans multiple orders of magnitude)
    ax.legend(title='Quantization Format', fontsize=10, title_fontsize=11)
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Scatter plot saved to: {output_path}")


def plot_alpha_mse_by_category(
    df: pd.DataFrame,
    quant_formats: List[str],
    output_path: str
):
    """Create separate scatter plots for each layer category."""
    categories = df['category'].unique()
    n_cats = len(categories)
    
    fig, axes = plt.subplots(1, n_cats, figsize=(6*n_cats, 5))
    if n_cats == 1:
        axes = [axes]
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(quant_formats)))
    color_map = dict(zip(quant_formats, colors))
    
    for idx, category in enumerate(categories):
        ax = axes[idx]
        cat_data = df[df['category'] == category]
        
        for fmt in quant_formats:
            fmt_data = cat_data[cat_data['quantization_format'] == fmt]
            fmt_data = fmt_data.dropna(subset=['alpha', 'mse'])
            
            if len(fmt_data) > 0:
                ax.scatter(
                    fmt_data['alpha'],
                    fmt_data['mse'],
                    color=color_map[fmt],
                    label=fmt,
                    alpha=0.6,
                    s=40,
                    edgecolors='black',
                    linewidth=0.5
                )
        
        ax.set_xlabel('Alpha-Hill', fontsize=11, fontweight='bold')
        ax.set_ylabel('MSE', fontsize=11, fontweight='bold')
        ax.set_title(f'{category}', fontsize=12, fontweight='bold')
        ax.set_yscale('log')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Category scatter plot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze relationship between Alpha-Hill and quantization MSE"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HF model id or local path (e.g., meta-llama/Llama-3.1-8B)"
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
        default="mxfp8,mxfp4,fp8,fp4,int8,int6,int4,int3,int2",
        help="Comma-separated list of quantization formats to test"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results/alpha_mse_analysis",
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
    parser.add_argument(
        "--max-layers",
        type=int,
        default=None,
        help="Maximum number of layers to analyze (for quick testing)"
    )
    
    args = parser.parse_args()
    
    # Setup
    setup_logging(args.log_level)
    quant_formats = [f.strip() for f in args.quant_formats.split(',')]
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("ALPHA-HILL vs MSE ANALYSIS")
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
        
        # Check max layers limit
        if args.max_layers and layer_count > args.max_layers:
            print(f"\nReached max layers limit ({args.max_layers}), stopping...")
            break
        
        weight = module.weight.detach().contiguous()
        out_features, in_features = safe_get_in_out_features(module)
        category = categorize(name)
        
        print(f"[{layer_count}] {name}")
        print(f"    Shape: [{out_features}, {in_features}], Category: {category}")
        
        # Compute alpha and MSE for all formats
        alpha_original, mse_results = compute_alpha_and_mse_for_all_formats(
            weight, quant_formats, args.k_frac
        )
        
        print(f"    Alpha (original): {alpha_original:.4f}")
        
        # Print MSE for each format
        for fmt in quant_formats:
            mse = mse_results[fmt]
            if not np.isnan(mse):
                print(f"      {fmt:>6s}: MSE = {mse:.6e}")
            else:
                print(f"      {fmt:>6s}: FAILED")
        
        # Store results (one row per layer per quantization format)
        for fmt in quant_formats:
            result_row = {
                'layer_name': name,
                'category': category,
                'out_features': out_features,
                'in_features': in_features,
                'numel': int(weight.numel()),
                'alpha': alpha_original,
                'quantization_format': fmt,
                'mse': mse_results[fmt]
            }
            all_results.append(result_row)
        
        print()
    
    print("="*80)
    print(f"Analyzed {layer_count} layers")
    print("="*80 + "\n")
    
    # Save CSV
    df = pd.DataFrame(all_results)
    csv_path = output_dir / "alpha_mse_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"✓ Results saved to: {csv_path}")
    
    # Plot scatter plot
    plot_path = output_dir / "alpha_mse_scatter.png"
    plot_alpha_mse_scatter(df, quant_formats, str(plot_path))
    
    # Plot by category
    plot_cat_path = output_dir / "alpha_mse_by_category.png"
    plot_alpha_mse_by_category(df, quant_formats, str(plot_cat_path))
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80 + "\n")
    
    print("Alpha-Hill statistics (original weights):")
    alpha_values = df.drop_duplicates(subset=['layer_name'])['alpha'].dropna()
    if len(alpha_values) > 0:
        print(f"  Mean: {alpha_values.mean():.4f}")
        print(f"  Std:  {alpha_values.std():.4f}")
        print(f"  Min:  {alpha_values.min():.4f}")
        print(f"  Max:  {alpha_values.max():.4f}")
    
    print("\nMSE statistics by quantization format:")
    for fmt in quant_formats:
        fmt_data = df[df['quantization_format'] == fmt]['mse'].dropna()
        if len(fmt_data) > 0:
            print(f"  {fmt:>6s}: mean={fmt_data.mean():.6e}, "
                  f"std={fmt_data.std():.6e}, "
                  f"min={fmt_data.min():.6e}, "
                  f"max={fmt_data.max():.6e}")
    
    # Correlation analysis
    print("\n" + "="*80)
    print("CORRELATION ANALYSIS (Alpha vs MSE)")
    print("="*80 + "\n")
    
    for fmt in quant_formats:
        fmt_data = df[df['quantization_format'] == fmt][['alpha', 'mse']].dropna()
        if len(fmt_data) > 10:  # Need sufficient data points
            correlation = fmt_data['alpha'].corr(fmt_data['mse'])
            print(f"  {fmt:>6s}: correlation = {correlation:.4f}")
    
    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)
    print(f"\nOutput directory: {output_dir}")
    print(f"  - CSV: {csv_path.name}")
    print(f"  - Main scatter plot: alpha_mse_scatter.png")
    print(f"  - Category scatter plot: alpha_mse_by_category.png")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
