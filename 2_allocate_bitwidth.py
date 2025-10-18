#!/usr/bin/env python
"""
Step 2: Allocate bitwidth based on Alpha-Hill values

This script automatically assigns quantization precision to each layer
based on Alpha-Hill sensitivity values. Layers with higher alpha values
(more sensitive) get higher precision.

By default, the following layers are NOT quantized as they are critical:
- Attention layers: q_proj, k_proj, v_proj, o_proj
- Routing gate/router layers (NOT gate_proj/up_proj/down_proj)

Mixed precision allocation (mxfp4-ratio) is applied to remaining layers,
mainly: gate_proj, up_proj, down_proj in MoE FFN blocks.

Usage:
    # Basic usage (skips attention and routing gate layers by default)
    # Applies mixed precision to gate_proj/up_proj/down_proj
    python 2_allocate_bitwidth.py \\
        --model allenai/OLMoE-1B-7B-0924 \\
        --alpha-csv results/alpha_values.csv \\
        --mxfp4-ratio 0.3 \\
        --output configs/auto_quant_config.json
    
    # Quantize everything including attention layers
    python 2_allocate_bitwidth.py \\
        --model allenai/OLMoE-1B-7B-0924 \\
        --alpha-csv results/alpha_values.csv \\
        --no-skip-attention \\
        --no-skip-gate \\
        --output configs/aggressive_quant.json

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
    bf16_ratio: float = 0.0,
    skip_attention: bool = True,
    skip_gate: bool = True
) -> Dict[str, Any]:
    """
    Create quantization config based on Alpha-Hill values.
    
    Args:
        alpha_results: Dict mapping layer names to alpha values
        mxfp4_ratio: Ratio of layers to use mxfp4 (high precision for sensitive layers)
        bf16_ratio: Ratio of layers to keep in bf16 (skip quantization)
        skip_attention: Skip quantizing attention layers (q_proj, k_proj, v_proj, o_proj)
        skip_gate: Skip quantizing gate/router layers
    
    Returns:
        Quantization config dict
    """
    # Create config
    config = {
        "default": {
            "wq": "mxfp8",
            "aq": "mxfp8", 
            "group_size": 128
        },
        "overrides": []
    }
    
    # Skip attention layers by default (more sensitive to quantization)
    if skip_attention:
        attention_patterns = ["*.q_proj", "*.k_proj", "*.v_proj", "*.o_proj"]
        for pattern in attention_patterns:
            config["overrides"].append({
                "pattern": pattern,
                "skip": True,
                "comment": "Skip attention layer (sensitive)"
            })
    
    # Skip gate/router layers by default (critical for MoE routing)
    # Note: This skips routing gates like "*.gate" or "*.router", NOT gate_proj
    if skip_gate:
        gate_patterns = ["*.gate", "*.router"]
        for pattern in gate_patterns:
            config["overrides"].append({
                "pattern": pattern,
                "skip": True,
                "comment": "Skip gate/router layer (critical for routing)"
            })
    
    # Filter out attention and gate layers from alpha-based allocation
    # This ensures mxfp4-ratio only applies to remaining layers (gate_proj, up_proj, down_proj)
    filtered_layers = []
    for layer_name, alpha_value in alpha_results.items():
        name_lower = layer_name.lower()
        
        # Check if this is an attention layer
        is_attention = any(kw in name_lower for kw in ['q_proj', 'k_proj', 'v_proj', 'o_proj'])
        
        # Check if this is a routing gate/router layer (NOT gate_proj/up_proj/down_proj)
        # Skip only routing gates: ends with .gate or contains router (but not gate_proj)
        is_routing_gate = (name_lower.endswith('.gate') or 'router' in name_lower) and 'gate_proj' not in name_lower
        
        # Skip if we're excluding these layers
        if (skip_attention and is_attention) or (skip_gate and is_routing_gate):
            continue
        
        filtered_layers.append((layer_name, alpha_value))
    
    # Sort remaining layers by alpha value (descending)
    sorted_layers = sorted(filtered_layers, key=lambda x: x[1], reverse=True)
    total_layers = len(sorted_layers)
    
    # Calculate thresholds
    n_bf16 = int(total_layers * bf16_ratio)
    n_mxfp4 = int(total_layers * mxfp4_ratio)
    
    # Assign precision based on alpha values for remaining layers
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
        "--skip-attention",
        action="store_true",
        default=True,
        help="Skip quantizing attention layers (q_proj, k_proj, v_proj, o_proj) [default: True]"
    )
    parser.add_argument(
        "--no-skip-attention",
        dest="skip_attention",
        action="store_false",
        help="Quantize attention layers"
    )
    parser.add_argument(
        "--skip-gate",
        action="store_true",
        default=True,
        help="Skip quantizing gate/router layers [default: True]"
    )
    parser.add_argument(
        "--no-skip-gate",
        dest="skip_gate",
        action="store_false",
        help="Quantize gate/router layers"
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
    logger.info(f"  Skip attention layers: {args.skip_attention}")
    logger.info(f"  Skip gate/router layers: {args.skip_gate}")
    
    config = create_quantization_config(
        alpha_results,
        mxfp4_ratio=args.mxfp4_ratio,
        bf16_ratio=args.bf16_ratio,
        skip_attention=args.skip_attention,
        skip_gate=args.skip_gate
    )
    
    # Save config
    logger.info(f"Saving config to: {args.output}")
    with open(args.output, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Count skipped layers
    total_alpha_layers = len(alpha_results)
    skipped_attention = sum(1 for name in alpha_results.keys() 
                           if args.skip_attention and any(kw in name.lower() for kw in ['q_proj', 'k_proj', 'v_proj', 'o_proj']))
    # Only count routing gates (not gate_proj/up_proj/down_proj)
    skipped_gate = sum(1 for name in alpha_results.keys() 
                      if args.skip_gate and ((name.lower().endswith('.gate') or 'router' in name.lower()) and 'gate_proj' not in name.lower()))
    
    quantizable_layers = total_alpha_layers - skipped_attention - skipped_gate
    n_bf16 = int(quantizable_layers * args.bf16_ratio)
    n_mxfp4 = int(quantizable_layers * args.mxfp4_ratio)
    n_mxfp8 = quantizable_layers - n_bf16 - n_mxfp4
    
    logger.info(f"✓ Bitwidth allocation complete!")
    logger.info(f"  Total layers analyzed: {total_alpha_layers}")
    if args.skip_attention:
        logger.info(f"  Skipped attention layers (q/k/v/o_proj): {skipped_attention}")
    if args.skip_gate:
        logger.info(f"  Skipped routing gate/router layers: {skipped_gate}")
    logger.info(f"  Quantizable layers (mainly gate_proj/up_proj/down_proj): {quantizable_layers}")
    logger.info(f"    - bf16 layers: {n_bf16}")
    logger.info(f"    - mxfp4 layers: {n_mxfp4}")
    logger.info(f"    - mxfp8 layers: {n_mxfp8}")
    logger.info(f"\nNext step: Use 3_gptq_quantize.py to quantize the model")


if __name__ == "__main__":
    main()

