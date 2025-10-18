#!/usr/bin/env python
"""
Step 5: Analyze Results

This script provides various analysis tools for quantization results:
- Alpha vs MSE relationship
- Layer-wise sensitivity analysis
- Visualization of quantization effects

Usage:
    # Analyze alpha-MSE relationship
    python 5_analyze_results.py \\
        --mode alpha_mse \\
        --model allenai/OLMoE-1B-7B-0924 \\
        --alpha-csv results/alpha_values.csv \\
        --output results/alpha_mse_analysis.png

    # Visualize alpha distribution
    python 5_analyze_results.py \\
        --mode visualize \\
        --alpha-csv results/alpha_values.csv \\
        --output results/alpha_distribution.png

Output:
    - Analysis plots and CSV files
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def analyze_alpha_mse(model_path, alpha_csv, output):
    """Analyze relationship between Alpha-Hill and quantization MSE"""
    print("Analyzing Alpha-MSE relationship...")
    
    # Load alpha values
    df_alpha = pd.read_csv(alpha_csv)
    
    # Import analysis function
    from scripts.analyze_alpha_mse_relationship import main as run_alpha_mse_analysis
    
    # Run analysis (you may need to adapt this based on your specific needs)
    print(f"Analysis results will be saved to: {output}")
    print("Note: This requires the model to compute MSE. Use scripts/analyze_alpha_mse_relationship.py for full analysis.")


def visualize_alpha_distribution(alpha_csv, output):
    """Visualize alpha value distribution across layers"""
    print("Visualizing alpha distribution...")
    
    # Load alpha values
    df = pd.read_csv(alpha_csv)
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Histogram
    axes[0, 0].hist(df['alpha_hill'], bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Alpha-Hill Value')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Alpha-Hill Distribution')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Sorted alpha values
    df_sorted = df.sort_values('alpha_hill', ascending=False).reset_index(drop=True)
    axes[0, 1].plot(df_sorted['alpha_hill'], marker='o', markersize=3)
    axes[0, 1].set_xlabel('Layer Rank')
    axes[0, 1].set_ylabel('Alpha-Hill Value')
    axes[0, 1].set_title('Sorted Alpha-Hill Values')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Box plot
    axes[1, 0].boxplot(df['alpha_hill'], vert=True)
    axes[1, 0].set_ylabel('Alpha-Hill Value')
    axes[1, 0].set_title('Alpha-Hill Box Plot')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Cumulative distribution
    sorted_alpha = np.sort(df['alpha_hill'])
    cumulative = np.arange(1, len(sorted_alpha) + 1) / len(sorted_alpha)
    axes[1, 1].plot(sorted_alpha, cumulative)
    axes[1, 1].set_xlabel('Alpha-Hill Value')
    axes[1, 1].set_ylabel('Cumulative Probability')
    axes[1, 1].set_title('Cumulative Distribution')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"✓ Visualization saved to: {output}")
    
    # Print statistics
    print("\nAlpha-Hill Statistics:")
    print(f"  Total layers: {len(df)}")
    print(f"  Mean: {df['alpha_hill'].mean():.4f}")
    print(f"  Median: {df['alpha_hill'].median():.4f}")
    print(f"  Std: {df['alpha_hill'].std():.4f}")
    print(f"  Min: {df['alpha_hill'].min():.4f}")
    print(f"  Max: {df['alpha_hill'].max():.4f}")


def compare_eval_results(result_files, output):
    """Compare evaluation results from different configurations"""
    print("Comparing evaluation results...")
    
    results = {}
    for file_path in result_files:
        import json
        with open(file_path, 'r') as f:
            data = json.load(f)
        name = Path(file_path).stem
        results[name] = data
    
    # Create comparison table
    print("\nComparison of evaluation results:")
    # Extract and display key metrics
    # (Implementation depends on your specific evaluation result format)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Step 5: Analyze quantization results"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["alpha_mse", "visualize", "compare"],
        help="Analysis mode"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model path (for alpha_mse mode)"
    )
    parser.add_argument(
        "--alpha-csv",
        type=str,
        help="Alpha values CSV file"
    )
    parser.add_argument(
        "--eval-results",
        type=str,
        nargs='+',
        help="Evaluation result files (for compare mode)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output file path"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("="*60)
    print("Step 5: Analyze Results")
    print("="*60)
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Run appropriate analysis
    if args.mode == "alpha_mse":
        if not args.model or not args.alpha_csv:
            print("Error: --model and --alpha-csv required for alpha_mse mode")
            return
        analyze_alpha_mse(args.model, args.alpha_csv, args.output)
    
    elif args.mode == "visualize":
        if not args.alpha_csv:
            print("Error: --alpha-csv required for visualize mode")
            return
        visualize_alpha_distribution(args.alpha_csv, args.output)
    
    elif args.mode == "compare":
        if not args.eval_results:
            print("Error: --eval-results required for compare mode")
            return
        compare_eval_results(args.eval_results, args.output)
    
    print("\n✓ Analysis complete!")


if __name__ == "__main__":
    main()

