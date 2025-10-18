#!/usr/bin/env python
"""
Step 2: Allocate bitwidth based on Alpha-Hill values

This script automatically assigns quantization precision to each layer
based on Alpha-Hill sensitivity values. Layers with higher alpha values
(more sensitive) get higher precision.

Usage:
    python 2_allocate_bitwidth.py \\
        --model allenai/OLMoE-1B-7B-0924 \\
        --alpha-csv results/alpha_values.csv \\
        --mxfp4-ratio 0.3 \\
        --output configs/auto_quant_config.json

Output:
    - JSON config file with layer-wise quantization settings
    - Can be used for GPTQ quantization in step 3
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any

from alphaquant.alpha_hill.utils import alpha_hill_from_model, setup_logging
from alphaquant.utils.hf_utils import load_hf_causal_lm


def create_quantization_config(
    alpha_results: Dict[str, float],
    mxfp4_ratio: float = 0.3,
    bf16_ratio: float = 0.0
) -> Dict[str, Any]:
    """
    Create quantization config based on Alpha-Hill values.
    
    Args:
        alpha_results: Dict mapping layer names to alpha values
        mxfp4_ratio: Ratio of layers to use mxfp4 (high precision for sensitive layers)
        bf16_ratio: Ratio of layers to keep in bf16 (skip quantization)
    
    Returns:
        Quantization config dict
    """
    # Sort layers by alpha value (descending)
    sorted_layers = sorted(alpha_results.items(), key=lambda x: x[1], reverse=True)
    total_layers = len(sorted_layers)
    
    # Calculate thresholds
    n_bf16 = int(total_layers * bf16_ratio)
    n_mxfp4 = int(total_layers * mxfp4_ratio)
    
    # Create config
    config = {
        "default": {
            "wq": "mxfp8",
            "aq": "mxfp8", 
            "group_size": 128
        },
        "overrides": []
    }
    
    # Assign precision based on alpha values
    for idx, (layer_name, alpha_value) in enumerate(sorted_layers):
        if idx < n_bf16:
            # Most sensitive: keep in bf16
            config["overrides"].append({
                "pattern": layer_name,
                "skip": True,
                "comment": f"Top {bf16_ratio*100:.0f}% alpha, keep bf16"
            })
        elif idx < n_bf16 + n_mxfp4:
            # High sensitivity: use mxfp4
            config["overrides"].append({
                "pattern": layer_name,
                "wq": "mxfp4",
                "group_size": 64,
                "comment": f"High alpha, use mxfp4"
            })
        # Rest use default mxfp8
    
    return config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Step 2: Allocate bitwidth based on Alpha-Hill values"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model ID or local path"
    )
    parser.add_argument(
        "--alpha-csv",
        type=str,
        help="CSV file with Alpha-Hill values (from step 1)"
    )
    parser.add_argument(
        "--mxfp4-ratio",
        type=float,
        default=0.3,
        help="Ratio of layers to use mxfp4 (0.0-1.0)"
    )
    parser.add_argument(
        "--bf16-ratio",
        type=float,
        default=0.0,
        help="Ratio of layers to keep in bf16 (0.0-1.0)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="configs/auto_quant_config.json",
        help="Output JSON config file"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use (cpu/cuda)"
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
    
    logger = setup_logging(args.log_level)
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load or compute Alpha-Hill values
    if args.alpha_csv:
        logger.info(f"Loading Alpha-Hill values from: {args.alpha_csv}")
        df = pd.read_csv(args.alpha_csv)
        alpha_results = dict(zip(df['layer_name'], df['alpha_hill']))
    else:
        logger.info(f"Computing Alpha-Hill values for model: {args.model}")
        model = load_hf_causal_lm(
            model_name=args.model,
            device=args.device,
            dtype="fp32"
        )
        alpha_results = alpha_hill_from_model(model)
    
    # Create quantization config
    logger.info("Creating quantization configuration...")
    logger.info(f"  mxfp4 ratio: {args.mxfp4_ratio}")
    logger.info(f"  bf16 ratio: {args.bf16_ratio}")
    
    config = create_quantization_config(
        alpha_results,
        mxfp4_ratio=args.mxfp4_ratio,
        bf16_ratio=args.bf16_ratio
    )
    
    # Save config
    logger.info(f"Saving config to: {args.output}")
    with open(args.output, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"✓ Bitwidth allocation complete!")
    logger.info(f"  Total layers: {len(alpha_results)}")
    logger.info(f"  bf16 layers: {int(len(alpha_results) * args.bf16_ratio)}")
    logger.info(f"  mxfp4 layers: {int(len(alpha_results) * args.mxfp4_ratio)}")
    logger.info(f"  mxfp8 layers: {len(alpha_results) - int(len(alpha_results) * (args.bf16_ratio + args.mxfp4_ratio))}")
    logger.info(f"\nNext step: Use 3_gptq_quantize.py to quantize the model")


if __name__ == "__main__":
    main()

