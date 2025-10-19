#!/usr/bin/env python
"""
Step 5: Analyze and Compare GSM8K Results

This script loads results from all experiments and creates comparison tables and analysis.

Usage:
    python gsm8k_analysis/5_analyze_results.py \\
        --baseline gsm8k_analysis/results/baseline.json \\
        --mxfp4_rtn gsm8k_analysis/results/mxfp4_rtn.json \\
        --gptq_wikitext2 gsm8k_analysis/results/gptq_mxfp4_wikitext2.json \\
        --gptq_gsm8k gsm8k_analysis/results/gptq_mxfp4_gsm8k.json \\
        --output gsm8k_analysis/results/comparison.csv
"""

import sys
import os
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json
import pandas as pd
from typing import Dict, Any


def extract_accuracy(result_file: Path) -> Dict[str, Any]:
    """Extract accuracy metrics from a result JSON file."""
    with open(result_file, 'r') as f:
        data = json.load(f)
    
    # Handle different result formats
    results = data.get('results', data)
    
    # Extract GSM8K results
    if 'results' in results and 'gsm8k' in results['results']:
        gsm8k_results = results['results']['gsm8k']
    elif 'gsm8k' in results:
        gsm8k_results = results['gsm8k']
    else:
        return {"error": "No GSM8K results found"}
    
    # Extract metrics
    metrics = {}
    
    # Try different accuracy metric names
    for key in ['exact_match,strict-match', 'exact_match', 'acc']:
        if key in gsm8k_results:
            metrics['accuracy'] = gsm8k_results[key]
            break
    
    # Extract stderr if available
    for key in ['exact_match,strict-match_stderr', 'exact_match_stderr', 'acc_stderr']:
        if key in gsm8k_results:
            metrics['stderr'] = gsm8k_results[key]
            break
    
    # Add metadata
    if 'calibration_data' in data:
        metrics['calibration_data'] = data['calibration_data']
    if 'num_quantized_layers' in data:
        metrics['num_quantized_layers'] = data['num_quantized_layers']
    if 'quantization_format' in data:
        metrics['quantization_format'] = data['quantization_format']
    
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description="Step 5: Analyze GSM8K quantization results"
    )
    
    parser.add_argument(
        "--baseline",
        type=str,
        default="gsm8k_analysis/results/baseline.json",
        help="Path to baseline results"
    )
    parser.add_argument(
        "--mxfp4_rtn",
        type=str,
        default="gsm8k_analysis/results/mxfp4_rtn.json",
        help="Path to MXFP4 (RTN) results"
    )
    parser.add_argument(
        "--gptq_wikitext2",
        type=str,
        default="gsm8k_analysis/results/gptq_mxfp4_wikitext2.json",
        help="Path to GPTQ + MXFP4 (WikiText2) results"
    )
    parser.add_argument(
        "--gptq_gsm8k",
        type=str,
        default="gsm8k_analysis/results/gptq_mxfp4_gsm8k.json",
        help="Path to GPTQ + MXFP4 (GSM8K) results"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="gsm8k_analysis/results/comparison.csv",
        help="Output file for comparison table"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 70)
    print("Step 5: GSM8K Results Analysis")
    print("=" * 70)
    
    # Define experiments
    experiments = {
        'baseline': args.baseline,
        'mxfp4_rtn': args.mxfp4_rtn,
        'gptq_mxfp4_wikitext2': args.gptq_wikitext2,
        'gptq_mxfp4_gsm8k': args.gptq_gsm8k
    }
    
    # Extract results from all experiments
    comparison_data = []
    
    for exp_name, result_file in experiments.items():
        result_path = Path(result_file)
        
        if not result_path.exists():
            print(f"\n⚠ File not found: {result_file}")
            continue
        
        print(f"\nProcessing {exp_name}...")
        metrics = extract_accuracy(result_path)
        
        if 'error' in metrics:
            print(f"  ⚠ {metrics['error']}")
            continue
        
        # Create row for comparison
        row = {
            'Experiment': exp_name,
            'Accuracy': metrics.get('accuracy', 'N/A'),
            'Stderr': metrics.get('stderr', 'N/A'),
            'Method': metrics.get('quantization_format', 'N/A'),
            'Calibration': metrics.get('calibration_data', 'N/A'),
            'Quantized Layers': metrics.get('num_quantized_layers', 'N/A')
        }
        
        comparison_data.append(row)
        if isinstance(metrics.get('accuracy'), float):
            print(f"  ✓ Accuracy: {metrics.get('accuracy'):.4f}")
        else:
            print(f"  ✓ Accuracy: {metrics.get('accuracy')}")
    
    # Create comparison DataFrame
    df = pd.DataFrame(comparison_data)
    
    # Sort by accuracy (descending)
    if 'Accuracy' in df.columns and df['Accuracy'].dtype in [float, int]:
        df = df.sort_values('Accuracy', ascending=False)
    
    # Print comparison table
    print("\n" + "=" * 70)
    print("Comparison Table")
    print("=" * 70)
    print(df.to_string(index=False))
    
    # Calculate accuracy drops
    if len(df) > 0 and 'baseline' in df['Experiment'].values:
        baseline_acc = df[df['Experiment'] == 'baseline']['Accuracy'].values[0]
        
        print("\n" + "=" * 70)
        print("Accuracy Drop from Baseline")
        print("=" * 70)
        
        for _, row in df.iterrows():
            if row['Experiment'] != 'baseline':
                acc = row['Accuracy']
                if isinstance(acc, (float, int)) and isinstance(baseline_acc, (float, int)):
                    drop = baseline_acc - acc
                    drop_pct = (drop / baseline_acc) * 100
                    print(f"{row['Experiment']:30s}: {drop:+.4f} ({drop_pct:+.2f}%)")
    
    # Save comparison table
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"\n✓ Comparison table saved to: {output_path}")
    
    # Save detailed analysis
    analysis_output = output_path.with_suffix('.txt')
    with open(analysis_output, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("GSM8K Quantization Analysis\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("=" * 70 + "\n")
        f.write("Results\n")
        f.write("=" * 70 + "\n\n")
        f.write(df.to_string(index=False))
        f.write("\n\n")
        
        if len(df) > 0 and 'baseline' in df['Experiment'].values:
            baseline_acc = df[df['Experiment'] == 'baseline']['Accuracy'].values[0]
            
            f.write("=" * 70 + "\n")
            f.write("Accuracy Drop from Baseline\n")
            f.write("=" * 70 + "\n\n")
            
            for _, row in df.iterrows():
                if row['Experiment'] != 'baseline':
                    acc = row['Accuracy']
                    if isinstance(acc, (float, int)) and isinstance(baseline_acc, (float, int)):
                        drop = baseline_acc - acc
                        drop_pct = (drop / baseline_acc) * 100
                        f.write(f"{row['Experiment']:30s}: {drop:+.4f} ({drop_pct:+.2f}%)\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("Key Findings\n")
        f.write("=" * 70 + "\n\n")
        
        if len(df) > 1:
            # Find best quantization method
            quant_methods = df[df['Experiment'] != 'baseline']
            if len(quant_methods) > 0:
                best = quant_methods.iloc[0]
                f.write(f"Best quantization method: {best['Experiment']}\n")
                f.write(f"  Accuracy: {best['Accuracy']:.4f}\n")
                if 'Method' in best and best['Method'] != 'N/A':
                    f.write(f"  Method: {best['Method']}\n")
                if 'Calibration' in best and best['Calibration'] != 'N/A':
                    f.write(f"  Calibration: {best['Calibration']}\n")
                
                # Compare GPTQ methods if both exist
                gptq_methods = df[df['Experiment'].str.contains('gptq')]
                if len(gptq_methods) >= 2:
                    f.write("\nGPTQ + MXFP4 Calibration Comparison:\n")
                    for _, row in gptq_methods.iterrows():
                        f.write(f"  {row['Calibration']:10s}: {row['Accuracy']:.4f}\n")
    
    print(f"✓ Detailed analysis saved to: {analysis_output}")
    print("\n✓ Step 5 complete!")


if __name__ == '__main__':
    main()

