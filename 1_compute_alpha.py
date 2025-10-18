#!/usr/bin/env python
"""
Step 1: Compute Alpha-Hill values for model layers

This script computes Alpha-Hill values for each layer in the model.
Alpha-Hill is a quantization sensitivity metric that helps determine
which layers are more sensitive to quantization.

Usage:
    python 1_compute_alpha.py \\
        --model allenai/OLMoE-1B-7B-0924 \\
        --output results/alpha_values.csv \\
        --device cuda

Output:
    - CSV file with Alpha-Hill values for each layer
    - Can be used for automatic bitwidth allocation in step 2
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import json
import pandas as pd
from pathlib import Path

from alphaquant.alpha_hill.utils import alpha_hill_from_model, setup_logging
from alphaquant.utils.hf_utils import load_hf_causal_lm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Step 1: Compute Alpha-Hill values for model layers"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        required=True, 
        help="HuggingFace model ID or local path"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="results/alpha_values.csv",
        help="Output CSV file path for Alpha-Hill values"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cpu",
        help="Device to use (cpu/cuda)"
    )
    parser.add_argument(
        "--dtype", 
        type=str, 
        default="fp32",
        choices=["fp32", "fp16", "bf16"],
        help="Model dtype"
    )
    parser.add_argument(
        "--log-level", 
        type=str, 
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading model: {args.model}")
    logger.info(f"Device: {args.device}, dtype: {args.dtype}")
    
    # Load model
    model = load_hf_causal_lm(
        model_name=args.model,
        device=args.device,
        dtype=args.dtype
    )
    
    # Compute Alpha-Hill values
    logger.info("Computing Alpha-Hill values...")
    alpha_results = alpha_hill_from_model(model)
    
    # Save results
    logger.info(f"Saving results to: {args.output}")
    
    # Convert to DataFrame for easier viewing
    df_data = []
    for name, value in alpha_results.items():
        df_data.append({
            "layer_name": name,
            "alpha_hill": value
        })
    
    df = pd.DataFrame(df_data)
    df = df.sort_values("alpha_hill", ascending=False)
    df.to_csv(args.output, index=False)
    
    logger.info(f"✓ Alpha-Hill computation complete!")
    logger.info(f"  Total layers: {len(alpha_results)}")
    logger.info(f"  Alpha range: [{df['alpha_hill'].min():.4f}, {df['alpha_hill'].max():.4f}]")
    logger.info(f"\nNext step: Use 2_allocate_bitwidth.py to assign quantization precision")


if __name__ == "__main__":
    main()

