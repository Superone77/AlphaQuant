#!/usr/bin/env python
from __future__ import annotations
import argparse
import json
import os
import sys
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alphaquant.alpha_hill.utils import alpha_hill_from_model, setup_logging
from alphaquant.utils.hf_utils import load_hf_causal_lm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantize model based on Alpha_Hill values")
    parser.add_argument("--model", type=str, required=True, help="HF model id or local path")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument("--dtype", type=str, default="fp32", help="fp32|fp16|bf16")
    parser.add_argument("--mxfp4-ratio", type=float, default=0.3, 
                       help="Ratio of layers to use mxfp4 (0.0-1.0)")
    parser.add_argument("--alpha-threshold", type=float, default=None,
                       help="Manual alpha threshold (if not provided, uses percentile)")
    parser.add_argument("--output-config", type=str, required=True,
                       help="Path to save the generated quantization config")
    parser.add_argument("--output-csv", type=str, default=None,
                       help="Optional path to save Alpha_Hill results as CSV")
    parser.add_argument("--log-level", type=str, default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                       help="Logging level")
    return parser.parse_args()


def create_quantization_config(alpha_results: Dict[str, Any], 
                              mxfp4_ratio: float = 0.3,
                              alpha_threshold: float = None) -> Dict[str, Any]:
    """Create quantization configuration based on Alpha_Hill values.
    
    Args:
        alpha_results: Dictionary of Alpha_Hill results from alpha_hill_from_model
        mxfp4_ratio: Target ratio of layers to use mxfp4
        alpha_threshold: Manual threshold for mxfp4 vs mxfp8 decision
        
    Returns:
        Quantization configuration dictionary
    """
    # Filter out failed computations and extract valid alpha values
    valid_results = {}
    for name, result in alpha_results.items():
        if isinstance(result.get('alpha'), (int, float)) and np.isfinite(result['alpha']):
            valid_results[name] = result
    
    if not valid_results:
        raise ValueError("No valid Alpha_Hill values found")
    
    # Sort layers by alpha value (descending)
    sorted_layers = sorted(valid_results.items(), 
                          key=lambda x: x[1]['alpha'], reverse=True)
    
    # Determine threshold
    if alpha_threshold is not None:
        threshold = alpha_threshold
    else:
        # Use percentile-based threshold
        alpha_values = [result['alpha'] for result in valid_results.values()]
        threshold = np.percentile(alpha_values, (1 - mxfp4_ratio) * 100)
    
    # Create configuration
    config = {
        "default": {
            "wq": "mxfp8",
            "aq": "mxfp8", 
            "group_size": 32
        },
        "overrides": []
    }
    
    # Add mxfp4 layers (high alpha)
    mxfp4_count = 0
    total_count = len(sorted_layers)
    
    for name, result in sorted_layers:
        if "lm_head" in name or "mlp.gate" in name:
            config["overrides"].append({
                "pattern": name,
                "skip": True
            })
            continue
        if result['alpha'] >= threshold:
            # Use mxfp4 for high alpha layers
            config["overrides"].append({
                "pattern": name,
                "wq": "mxfp4",
                "aq": "mxfp4",
                "group_size": 32,
                "alpha_hill": result['alpha'],
                "category": result['category']
            })
            mxfp4_count += 1
        else:
            # Use mxfp8 for low alpha layers
            config["overrides"].append({
                "pattern": name,
                "wq": "mxfp8",
                "aq": "mxfp8",
                "group_size": 128,
                "alpha_hill": result['alpha'],
                "category": result['category']
            })
    
    # Add summary to config
    config["summary"] = {
        "total_layers": total_count,
        "mxfp4_layers": mxfp4_count,
        "mxfp8_layers": total_count - mxfp4_count,
        "mxfp4_ratio": mxfp4_count / total_count,
        "alpha_threshold": threshold,
        "alpha_stats": {
            "min": min(r['alpha'] for r in valid_results.values()),
            "max": max(r['alpha'] for r in valid_results.values()),
            "mean": np.mean([r['alpha'] for r in valid_results.values()]),
            "median": np.median([r['alpha'] for r in valid_results.values()])
        }
    }
    
    return config


def save_alpha_results_csv(alpha_results: Dict[str, Any], output_path: str) -> None:
    """Save Alpha_Hill results to CSV file."""
    rows = []
    for name, result in alpha_results.items():
        row = {
            'name': name,
            'category': result.get('category', 'unknown'),
            'shape': f"[{result.get('out_features', 'N/A')}, {result.get('in_features', 'N/A')}]",
            'out_features': result.get('out_features', 'N/A'),
            'in_features': result.get('in_features', 'N/A'),
            'numel': result.get('numel', 'N/A'),
            'dtype': result.get('dtype', 'N/A'),
            'device': result.get('device', 'N/A'),
            'alpha_hill': result.get('alpha', 'N/A'),
            'k_used': result.get('k_used', 'N/A'),
            'n_eigs': result.get('n_eigs', 'N/A'),
            'method': result.get('method', 'N/A'),
            'error': result.get('error', '')
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Alpha_Hill results saved to: {output_path}")


def print_summary(alpha_results: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Print summary of Alpha_Hill analysis and quantization plan."""
    summary = config["summary"]
    
    print("\n" + "="*60)
    print("ALPHA_HILL ANALYSIS SUMMARY")
    print("="*60)
    print(f"Total layers analyzed: {summary['total_layers']}")
    print(f"Layers using mxfp4: {summary['mxfp4_layers']} ({summary['mxfp4_ratio']:.1%})")
    print(f"Layers using mxfp8: {summary['mxfp8_layers']} ({summary['mxfp8_layers']/summary['total_layers']:.1%})")
    print(f"Alpha threshold: {summary['alpha_threshold']:.4f}")
    
    print(f"\nAlpha statistics:")
    stats = summary['alpha_stats']
    print(f"  Min: {stats['min']:.4f}")
    print(f"  Max: {stats['max']:.4f}")
    print(f"  Mean: {stats['mean']:.4f}")
    print(f"  Median: {stats['median']:.4f}")
    
    # Show some examples
    print(f"\nExample mxfp4 layers (high alpha):")
    mxfp4_examples = [ov for ov in config["overrides"] if ov["wq"] == "mxfp4"][:5]
    for ex in mxfp4_examples:
        print(f"  {ex['pattern']}: alpha={ex['alpha_hill']:.4f}, category={ex['category']}")
    
    print(f"\nExample mxfp8 layers (low alpha):")
    mxfp8_examples = [ov for ov in config["overrides"] if ov["wq"] == "mxfp8"][:5]
    for ex in mxfp8_examples:
        print(f"  {ex['pattern']}: alpha={ex['alpha_hill']:.4f}, category={ex['category']}")


def main() -> None:
    args = parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    print(f"Loading model: {args.model}")
    print(f"Target mxfp4 ratio: {args.mxfp4_ratio:.1%}")
    
    # Load model
    model = load_hf_causal_lm(args.model, device=args.device, dtype=args.dtype)
    model.eval()
    
    # Compute Alpha_Hill for all layers
    print("Computing Alpha_Hill values for all layers...")
    alpha_results = alpha_hill_from_model(
        model,
        k_frac=0.1,
        force_cpu_svd=True,
        use_lowrank=False
    )
    
    print(f"Computed Alpha_Hill for {len(alpha_results)} layers")
    
    # Create quantization configuration
    print("Creating quantization configuration...")
    config = create_quantization_config(
        alpha_results, 
        mxfp4_ratio=args.mxfp4_ratio,
        alpha_threshold=args.alpha_threshold
    )
    
    # Save configuration
    os.makedirs(os.path.dirname(args.output_config), exist_ok=True) if os.path.dirname(args.output_config) else None
    with open(args.output_config, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Quantization config saved to: {args.output_config}")
    
    # Save CSV if requested
    if args.output_csv:
        save_alpha_results_csv(alpha_results, args.output_csv)
    
    # Print summary
    print_summary(alpha_results, config)
    
    print(f"\nConfiguration file ready: {args.output_config}")
    print("You can now use this config with the quantize_model.py script:")


if __name__ == "__main__":
    main() 