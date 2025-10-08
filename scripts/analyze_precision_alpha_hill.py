#!/usr/bin/env python
"""
Analyze Alpha-Hill values across different precisions for each layer.
Generates bar charts and CSV output.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
from pathlib import Path
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze Alpha-Hill values across different precisions"
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
        "--load-dtype", 
        type=str, 
        default="fp32",
        help="Initial dtype to load model (fp32|fp16|bf16)"
    )
    parser.add_argument(
        "--precisions",
        type=str,
        default="fp32,fp16,bf16",
        help="Comma-separated list of precisions to test (fp32,fp16,bf16,fp64)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results/precision_analysis",
        help="Output directory for plots and CSV"
    )
    parser.add_argument(
        "--k-frac",
        type=float,
        default=0.1,
        help="Fraction of eigenvalues to use for Alpha-Hill computation"
    )
    parser.add_argument(
        "--filter-layers",
        type=str,
        default=None,
        help="Regex pattern to filter specific layers (optional)"
    )
    parser.add_argument(
        "--plot-format",
        type=str,
        default="png",
        choices=["png", "pdf", "svg"],
        help="Output format for plots"
    )
    parser.add_argument(
        "--max-layers-per-plot",
        type=int,
        default=10,
        help="Maximum number of layers per combined plot (0 for all in one)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level"
    )
    return parser.parse_args()


def dtype_str_to_torch(dtype_str: str) -> torch.dtype:
    """Convert dtype string to torch dtype."""
    dtype_map = {
        'fp32': torch.float32,
        'float32': torch.float32,
        'fp16': torch.float16,
        'float16': torch.float16,
        'bf16': torch.bfloat16,
        'bfloat16': torch.bfloat16,
        'fp64': torch.float64,
        'float64': torch.float64,
    }
    return dtype_map.get(dtype_str.lower(), torch.float32)


def compute_alpha_hill_multi_precision(
    weight: torch.Tensor,
    precisions: List[str],
    k_frac: float = 0.1,
    force_cpu_svd: bool = True
) -> Dict[str, Tuple[float, int, int, str]]:
    """
    Compute Alpha-Hill for a weight tensor at different precisions.
    
    Args:
        weight: Original weight tensor
        precisions: List of precision strings (e.g., ['fp32', 'fp16', 'bf16'])
        k_frac: Fraction of eigenvalues to use
        force_cpu_svd: Whether to force SVD on CPU
    
    Returns:
        Dictionary mapping precision to (alpha, k_used, n_eigs, method)
    """
    results = {}
    
    for precision in precisions:
        try:
            # Convert weight to target precision
            target_dtype = dtype_str_to_torch(precision)
            weight_converted = weight.to(dtype=target_dtype)
            
            # Compute Alpha-Hill
            alpha, k_used, n_eigs, method = alpha_hill_from_weight(
                weight_converted,
                k=None,
                k_frac=k_frac,
                eps=1e-12,
                use_lowrank=False,
                force_cpu_svd=force_cpu_svd,
                use_eig=False
            )
            
            results[precision] = (alpha, k_used, n_eigs, method)
            
        except Exception as e:
            print(f"Error computing alpha for precision {precision}: {e}")
            results[precision] = (float('nan'), -1, -1, 'failed')
    
    return results


def plot_layer_precision_comparison(
    layer_results: Dict[str, Dict[str, float]],
    output_path: str,
    title: str = "Alpha-Hill by Precision"
):
    """
    Create a bar chart comparing alpha values across precisions for multiple layers.
    
    Args:
        layer_results: {layer_name: {precision: alpha_value}}
        output_path: Path to save the plot
        title: Plot title
    """
    # Prepare data
    layers = list(layer_results.keys())
    precisions = list(next(iter(layer_results.values())).keys())
    
    # Create figure
    fig, ax = plt.subplots(figsize=(max(12, len(layers) * 0.8), 8))
    
    x = np.arange(len(layers))
    width = 0.8 / len(precisions)
    
    # Plot bars for each precision
    for i, precision in enumerate(precisions):
        alphas = [layer_results[layer].get(precision, np.nan) for layer in layers]
        offset = (i - len(precisions) / 2) * width + width / 2
        ax.bar(x + offset, alphas, width, label=precision, alpha=0.8)
    
    # Customize plot
    ax.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax.set_ylabel('Alpha-Hill Value', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([layer.split('.')[-1] if len(layer) > 30 else layer 
                        for layer in layers], rotation=45, ha='right')
    ax.legend(title='Precision', loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {output_path}")


def plot_single_layer_bars(
    layer_name: str,
    precision_alphas: Dict[str, float],
    output_path: str
):
    """
    Create a bar chart for a single layer showing alpha values at different precisions.
    """
    precisions = list(precision_alphas.keys())
    alphas = list(precision_alphas.values())
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(precisions)))
    bars = ax.bar(precisions, alphas, color=colors, alpha=0.8, edgecolor='black')
    
    # Add value labels on bars
    for bar, alpha in zip(bars, alphas):
        height = bar.get_height()
        if not np.isnan(height):
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                   f'{alpha:.4f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Precision', fontsize=12, fontweight='bold')
    ax.set_ylabel('Alpha-Hill Value', fontsize=12, fontweight='bold')
    ax.set_title(f'Alpha-Hill: {layer_name}', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_summary_statistics(df: pd.DataFrame, precisions: List[str]) -> pd.DataFrame:
    """Create summary statistics for alpha values across precisions."""
    summary_data = []
    
    for precision in precisions:
        col = f'alpha_{precision}'
        if col in df.columns:
            valid_values = df[col].dropna()
            if len(valid_values) > 0:
                summary_data.append({
                    'precision': precision,
                    'count': len(valid_values),
                    'mean': valid_values.mean(),
                    'std': valid_values.std(),
                    'min': valid_values.min(),
                    'median': valid_values.median(),
                    'max': valid_values.max(),
                    'q25': valid_values.quantile(0.25),
                    'q75': valid_values.quantile(0.75)
                })
    
    return pd.DataFrame(summary_data)


def plot_precision_distribution(
    df: pd.DataFrame,
    precisions: List[str],
    output_path: str
):
    """Create box plots showing alpha distribution for each precision."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    data_to_plot = []
    labels = []
    
    for precision in precisions:
        col = f'alpha_{precision}'
        if col in df.columns:
            valid_data = df[col].dropna()
            if len(valid_data) > 0:
                data_to_plot.append(valid_data.values)
                labels.append(precision)
    
    if data_to_plot:
        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
        
        # Customize box plot colors
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(labels)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_xlabel('Precision', fontsize=12, fontweight='bold')
        ax.set_ylabel('Alpha-Hill Value', fontsize=12, fontweight='bold')
        ax.set_title('Alpha-Hill Distribution Across Precisions', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved distribution plot: {output_path}")


def main():
    args = parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Parse precisions
    precisions = [p.strip() for p in args.precisions.split(',')]
    print(f"Analyzing precisions: {precisions}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    single_layer_plots_dir = plots_dir / "individual_layers"
    single_layer_plots_dir.mkdir(exist_ok=True)
    
    # Load model
    print(f"\nLoading model: {args.model}")
    print(f"Device: {args.device}, Initial dtype: {args.load_dtype}")
    model = load_hf_causal_lm(args.model, device=args.device, dtype=args.load_dtype)
    model.eval()
    
    # Collect all results
    all_results = []
    layer_count = 0
    
    print(f"\nComputing Alpha-Hill values across {len(precisions)} precisions...")
    print("=" * 80)
    
    # Iterate through layers
    for name, module in iter_linear_modules(model):
        # Apply filter if specified
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
        
        print(f"\n[{layer_count}] {name}")
        print(f"    Shape: [{out_features}, {in_features}], Category: {category}")
        
        # Compute alpha for all precisions
        precision_results = compute_alpha_hill_multi_precision(
            weight,
            precisions,
            k_frac=args.k_frac,
            force_cpu_svd=True
        )
        
        # Prepare result row
        result_row = {
            'layer_name': name,
            'category': category,
            'out_features': out_features,
            'in_features': in_features,
            'numel': int(weight.numel()),
        }
        
        # Add precision-specific results
        precision_alphas = {}
        for precision, (alpha, k_used, n_eigs, method) in precision_results.items():
            result_row[f'alpha_{precision}'] = alpha
            result_row[f'k_used_{precision}'] = k_used
            result_row[f'n_eigs_{precision}'] = n_eigs
            result_row[f'method_{precision}'] = method
            
            precision_alphas[precision] = alpha
            
            if not np.isnan(alpha):
                print(f"    {precision:>6s}: α = {alpha:.6f} (k={k_used}, n={n_eigs})")
            else:
                print(f"    {precision:>6s}: FAILED")
        
        all_results.append(result_row)
        
        # Plot individual layer
        plot_path = single_layer_plots_dir / f"layer_{layer_count:03d}_{name.replace('.', '_')}.{args.plot_format}"
        plot_single_layer_bars(name, precision_alphas, str(plot_path))
    
    print("\n" + "=" * 80)
    print(f"Completed analysis of {layer_count} layers")
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Save to CSV
    csv_path = output_dir / "precision_alpha_hill_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Results saved to: {csv_path}")
    
    # Create summary statistics
    summary_df = create_summary_statistics(df, precisions)
    summary_path = output_dir / "precision_summary_statistics.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"✓ Summary statistics saved to: {summary_path}")
    
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    
    # Create grouped plots
    print(f"\n✓ Creating grouped comparison plots...")
    
    # Group layers by category
    categories = df['category'].unique()
    for category in categories:
        category_df = df[df['category'] == category]
        if len(category_df) == 0:
            continue
        
        layer_results = {}
        for _, row in category_df.iterrows():
            layer_name = row['layer_name']
            precision_alphas = {p: row[f'alpha_{p}'] for p in precisions}
            layer_results[layer_name] = precision_alphas
        
        if layer_results:
            plot_path = plots_dir / f"category_{category}.{args.plot_format}"
            plot_layer_precision_comparison(
                layer_results,
                str(plot_path),
                title=f"Alpha-Hill by Precision - {category}"
            )
    
    # Create overall distribution plot
    dist_plot_path = plots_dir / f"alpha_distribution_by_precision.{args.plot_format}"
    plot_precision_distribution(df, precisions, str(dist_plot_path))
    
    # Create combined plot with batches
    if args.max_layers_per_plot > 0 and len(df) > args.max_layers_per_plot:
        num_batches = (len(df) + args.max_layers_per_plot - 1) // args.max_layers_per_plot
        for batch_idx in range(num_batches):
            start_idx = batch_idx * args.max_layers_per_plot
            end_idx = min((batch_idx + 1) * args.max_layers_per_plot, len(df))
            batch_df = df.iloc[start_idx:end_idx]
            
            layer_results = {}
            for _, row in batch_df.iterrows():
                layer_name = row['layer_name']
                precision_alphas = {p: row[f'alpha_{p}'] for p in precisions}
                layer_results[layer_name] = precision_alphas
            
            plot_path = plots_dir / f"combined_batch_{batch_idx + 1}.{args.plot_format}"
            plot_layer_precision_comparison(
                layer_results,
                str(plot_path),
                title=f"Alpha-Hill by Precision (Layers {start_idx + 1}-{end_idx})"
            )
    else:
        # Single combined plot
        layer_results = {}
        for _, row in df.iterrows():
            layer_name = row['layer_name']
            precision_alphas = {p: row[f'alpha_{p}'] for p in precisions}
            layer_results[layer_name] = precision_alphas
        
        plot_path = plots_dir / f"all_layers_combined.{args.plot_format}"
        plot_layer_precision_comparison(
            layer_results,
            str(plot_path),
            title="Alpha-Hill by Precision - All Layers"
        )
    
    print(f"\n✓ All plots saved to: {plots_dir}")
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print(f"  - CSV results: {csv_path.name}")
    print(f"  - Summary stats: {summary_path.name}")
    print(f"  - Individual plots: {len(list(single_layer_plots_dir.glob('*')))} files")
    print(f"  - Grouped plots: {len(list(plots_dir.glob('*.{}'.format(args.plot_format))))} files")
    print("=" * 80)


if __name__ == "__main__":
    main()

