#!/usr/bin/env python
"""
Utility script to analyze precision analysis results and generate quantization recommendations.
"""
from __future__ import annotations
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json


def load_results(csv_path: str) -> pd.DataFrame:
    """Load precision analysis results from CSV."""
    df = pd.read_csv(csv_path)
    return df


def compute_precision_statistics(df: pd.DataFrame, precisions: List[str]) -> pd.DataFrame:
    """Compute statistics about precision variations."""
    alpha_cols = [f'alpha_{p}' for p in precisions if f'alpha_{p}' in df.columns]
    
    # Compute statistics
    df['alpha_mean'] = df[alpha_cols].mean(axis=1)
    df['alpha_std'] = df[alpha_cols].std(axis=1)
    df['alpha_var'] = df[alpha_cols].var(axis=1)
    df['alpha_min'] = df[alpha_cols].min(axis=1)
    df['alpha_max'] = df[alpha_cols].max(axis=1)
    df['alpha_range'] = df['alpha_max'] - df['alpha_min']
    df['alpha_cv'] = df['alpha_std'] / df['alpha_mean']  # Coefficient of variation
    
    return df


def identify_sensitive_layers(
    df: pd.DataFrame,
    alpha_threshold: float = 3.0,
    variation_threshold: float = 0.01,
    cv_threshold: float = 0.005
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Identify layers that are sensitive to precision changes.
    
    Returns:
        high_alpha_layers: Layers with high alpha values
        high_variation_layers: Layers with high precision variation
        robust_layers: Layers robust to precision changes
    """
    # High alpha (precision-sensitive due to singular value concentration)
    high_alpha = df[df['alpha_mean'] >= alpha_threshold].copy()
    
    # High variation across precisions
    high_variation = df[
        (df['alpha_std'] >= variation_threshold) | 
        (df['alpha_cv'] >= cv_threshold)
    ].copy()
    
    # Robust layers (low alpha AND low variation)
    robust = df[
        (df['alpha_mean'] < alpha_threshold) & 
        (df['alpha_std'] < variation_threshold) &
        (df['alpha_cv'] < cv_threshold)
    ].copy()
    
    return high_alpha, high_variation, robust


def generate_quantization_recommendations(
    df: pd.DataFrame,
    alpha_threshold: float = 3.0,
    variation_threshold: float = 0.01
) -> Dict[str, List[str]]:
    """
    Generate quantization recommendations based on analysis.
    
    Returns:
        Dictionary mapping quantization strategies to layer names
    """
    recommendations = {
        'fp16_or_bf16': [],  # High precision needed
        'fp8_or_int8': [],   # Medium precision acceptable
        'int4_or_fp4': [],   # Low precision acceptable
    }
    
    for _, row in df.iterrows():
        layer_name = row['layer_name']
        alpha_mean = row['alpha_mean']
        alpha_std = row['alpha_std']
        
        # Decision logic
        if alpha_mean >= alpha_threshold or alpha_std >= variation_threshold:
            # Precision-sensitive: use higher precision
            recommendations['fp16_or_bf16'].append(layer_name)
        elif alpha_mean >= 2.5:
            # Medium sensitivity: use medium precision
            recommendations['fp8_or_int8'].append(layer_name)
        else:
            # Robust: can use aggressive quantization
            recommendations['int4_or_fp4'].append(layer_name)
    
    return recommendations


def print_summary(df: pd.DataFrame, precisions: List[str]):
    """Print summary of precision analysis."""
    print("\n" + "="*80)
    print("PRECISION ANALYSIS SUMMARY")
    print("="*80)
    
    print(f"\nTotal layers analyzed: {len(df)}")
    
    # Overall statistics
    print("\nAlpha-Hill Statistics (averaged across precisions):")
    print(f"  Mean: {df['alpha_mean'].mean():.4f} ± {df['alpha_mean'].std():.4f}")
    print(f"  Median: {df['alpha_mean'].median():.4f}")
    print(f"  Range: [{df['alpha_mean'].min():.4f}, {df['alpha_mean'].max():.4f}]")
    
    # Variation statistics
    print("\nPrecision Variation Statistics:")
    print(f"  Mean std dev: {df['alpha_std'].mean():.6f}")
    print(f"  Mean CV: {df['alpha_cv'].mean():.4f}")
    print(f"  Max range: {df['alpha_range'].max():.6f}")
    
    # Per-precision statistics
    print("\nPer-Precision Alpha Values:")
    for precision in precisions:
        col = f'alpha_{precision}'
        if col in df.columns:
            values = df[col].dropna()
            print(f"  {precision:>6s}: {values.mean():.4f} ± {values.std():.4f}")
    
    # Category breakdown
    print("\nBy Category:")
    category_stats = df.groupby('category').agg({
        'alpha_mean': ['count', 'mean', 'std'],
        'alpha_std': 'mean'
    }).round(4)
    print(category_stats.to_string())


def print_recommendations(
    recommendations: Dict[str, List[str]],
    df: pd.DataFrame
):
    """Print quantization recommendations."""
    print("\n" + "="*80)
    print("QUANTIZATION RECOMMENDATIONS")
    print("="*80)
    
    total = len(df)
    
    for strategy, layers in recommendations.items():
        count = len(layers)
        percentage = (count / total) * 100
        
        print(f"\n{strategy.upper().replace('_', ' ')} ({count} layers, {percentage:.1f}%):")
        
        if strategy == 'fp16_or_bf16':
            print("  → Precision-sensitive layers, use FP16 or BF16")
            print("  → High alpha or high precision variation")
        elif strategy == 'fp8_or_int8':
            print("  → Medium sensitivity, FP8 or INT8 recommended")
            print("  → Moderate alpha values, stable across precisions")
        else:
            print("  → Robust layers, INT4 or FP4 acceptable")
            print("  → Low alpha, minimal precision variation")
        
        # Show example layers
        if len(layers) > 0:
            print(f"\n  Example layers:")
            for layer in layers[:5]:
                row = df[df['layer_name'] == layer].iloc[0]
                print(f"    - {layer}")
                print(f"      α_mean={row['alpha_mean']:.4f}, α_std={row['alpha_std']:.6f}, category={row['category']}")
            
            if len(layers) > 5:
                print(f"    ... and {len(layers) - 5} more")


def export_quantization_config(
    recommendations: Dict[str, List[str]],
    output_path: str,
    default_scheme: str = "mxfp8"
):
    """Export quantization config in AlphaQuant JSON format."""
    config = {
        "default": {
            "wq": default_scheme,
            "aq": default_scheme,
            "group_size": 32
        },
        "overrides": []
    }
    
    # Map recommendations to quantization schemes
    scheme_map = {
        'fp16_or_bf16': ('mxfp8', 'mxfp8', 64),
        'fp8_or_int8': ('mxfp8', 'mxfp8', 32),
        'int4_or_fp4': ('mxfp4', 'mxfp4', 32),
    }
    
    for strategy, layers in recommendations.items():
        wq, aq, group_size = scheme_map[strategy]
        
        for layer in layers:
            config["overrides"].append({
                "pattern": layer,
                "wq": wq,
                "aq": aq,
                "group_size": group_size,
                "source": "precision_analysis"
            })
    
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✓ Quantization config exported to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze precision analysis results and generate recommendations"
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to precision_alpha_hill_results.csv"
    )
    parser.add_argument(
        "--precisions",
        type=str,
        default="fp32,fp16,bf16",
        help="Comma-separated list of precisions analyzed"
    )
    parser.add_argument(
        "--alpha-threshold",
        type=float,
        default=3.0,
        help="Alpha threshold for high-precision requirement"
    )
    parser.add_argument(
        "--variation-threshold",
        type=float,
        default=0.01,
        help="Std dev threshold for precision sensitivity"
    )
    parser.add_argument(
        "--export-config",
        type=str,
        default=None,
        help="Export quantization config to JSON file"
    )
    parser.add_argument(
        "--export-detailed-csv",
        type=str,
        default=None,
        help="Export detailed analysis with statistics to CSV"
    )
    
    args = parser.parse_args()
    
    # Load results
    print(f"\nLoading results from: {args.csv}")
    df = load_results(args.csv)
    
    precisions = [p.strip() for p in args.precisions.split(',')]
    
    # Compute statistics
    df = compute_precision_statistics(df, precisions)
    
    # Print summary
    print_summary(df, precisions)
    
    # Identify sensitive layers
    high_alpha, high_variation, robust = identify_sensitive_layers(
        df,
        alpha_threshold=args.alpha_threshold,
        variation_threshold=args.variation_threshold
    )
    
    print("\n" + "="*80)
    print("LAYER CATEGORIZATION")
    print("="*80)
    print(f"\nHigh Alpha Layers (α ≥ {args.alpha_threshold}): {len(high_alpha)}")
    print(f"High Variation Layers (σ ≥ {args.variation_threshold}): {len(high_variation)}")
    print(f"Robust Layers: {len(robust)}")
    
    # Generate recommendations
    recommendations = generate_quantization_recommendations(
        df,
        alpha_threshold=args.alpha_threshold,
        variation_threshold=args.variation_threshold
    )
    
    print_recommendations(recommendations, df)
    
    # Export config if requested
    if args.export_config:
        export_quantization_config(recommendations, args.export_config)
    
    # Export detailed CSV if requested
    if args.export_detailed_csv:
        df_export = df[[
            'layer_name', 'category', 'out_features', 'in_features',
            'alpha_mean', 'alpha_std', 'alpha_cv', 'alpha_min', 'alpha_max', 'alpha_range'
        ] + [f'alpha_{p}' for p in precisions if f'alpha_{p}' in df.columns]]
        
        df_export.to_csv(args.export_detailed_csv, index=False)
        print(f"\n✓ Detailed analysis exported to: {args.export_detailed_csv}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()

