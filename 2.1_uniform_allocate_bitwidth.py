#!/usr/bin/env python
"""
Step 2.1: Uniform Bitwidth Allocation for Experts

This script assigns quantization precision based on layer position:
- Front half experts: one precision level
- Back half experts: another precision level

This is a simple baseline strategy that doesn't rely on Alpha-Hill values.

By default, the following layers are NOT quantized as they are critical:
- Attention layers: q_proj, k_proj, v_proj, o_proj
- Routing gate/router layers (NOT gate_proj/up_proj/down_proj)

Usage:
    # Allocate front half experts to mxfp6, back half to mxfp4
    python 2.1_uniform_allocate_bitwidth.py \
        --model allenai/OLMoE-1B-7B-0924 \
        --front-precision mxfp6 \
        --back-precision mxfp4 \
        --output configs/uniform_quant_config.json
    
    # Use int quantizers with custom group sizes
    python 2.1_uniform_allocate_bitwidth.py \
        --model allenai/OLMoE-1B-7B-0924 \
        --front-precision int4 \
        --back-precision int3 \
        --front-group-size 128 \
        --back-group-size 128 \
        --output configs/uniform_int_quant.json
    
    # Quantize everything including attention layers
    python 2.1_uniform_allocate_bitwidth.py \
        --model allenai/OLMoE-1B-7B-0924 \
        --front-precision mxfp8 \
        --back-precision mxfp4 \
        --no-skip-attention \
        --no-skip-gate \
        --output configs/aggressive_uniform.json

Output:
    - JSON config file with layer-wise quantization settings
    - Can be used for GPTQ quantization in step 3
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

from alphaquant.utils.hf_utils import load_hf_causal_lm
from alphaquant.alpha_hill.utils import setup_logging


def get_expert_layers(model) -> List[Tuple[int, str]]:
    """
    Extract expert layer names with their layer indices.
    
    Args:
        model: HuggingFace model
    
    Returns:
        List of (layer_index, layer_name) tuples for expert layers
    """
    expert_layers = []
    
    for name, module in model.named_modules():
        # Only include expert FFN layers (gate_proj, up_proj, down_proj)
        # Exclude attention layers and routing gates
        name_lower = name.lower()
        
        # Check if this is an expert layer (gate_proj, up_proj, down_proj)
        is_expert = any(kw in name_lower for kw in ['gate_proj', 'up_proj', 'down_proj'])
        
        if is_expert:
            # Extract layer index from the name
            # Format is typically: model.layers.{idx}.mlp.experts.{expert_id}.{proj_type}
            # or model.layers.{idx}.{proj_type}
            parts = name.split('.')
            layer_idx = None
            for i, part in enumerate(parts):
                if part == 'layers' and i + 1 < len(parts):
                    try:
                        layer_idx = int(parts[i + 1])
                        break
                    except ValueError:
                        continue
            
            if layer_idx is not None:
                expert_layers.append((layer_idx, name))
    
    # Sort by layer index
    expert_layers.sort(key=lambda x: x[0])
    
    return expert_layers


def create_uniform_quantization_config(
    model,
    front_precision: str = "mxfp6",
    back_precision: str = "mxfp4",
    front_group_size: int = 128,
    back_group_size: int = 128,
    skip_attention: bool = True,
    skip_gate: bool = True,
    default_precision: str = "mxfp8",
    default_group_size: int = 128
) -> Dict[str, Any]:
    """
    Create quantization config with uniform allocation.
    
    Args:
        model: HuggingFace model
        front_precision: Precision for front half experts (e.g., "mxfp6", "int4")
        back_precision: Precision for back half experts (e.g., "mxfp4", "int3")
        front_group_size: Group size for front half experts
        back_group_size: Group size for back half experts
        skip_attention: Skip quantizing attention layers
        skip_gate: Skip quantizing gate/router layers
        default_precision: Default precision for non-expert layers
        default_group_size: Default group size
    
    Returns:
        Quantization config dict
    """
    # Create config
    config = {
        "default": {
            "wq": default_precision,
            "aq": default_precision,
            "group_size": default_group_size
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
    
    # Get expert layers
    expert_layers = get_expert_layers(model)
    
    if len(expert_layers) == 0:
        print("Warning: No expert layers found in model!")
        return config
    
    # Split into front and back halves
    mid_point = len(expert_layers) // 2
    front_half = expert_layers[:mid_point]
    back_half = expert_layers[mid_point:]
    
    # Assign front half experts
    for layer_idx, layer_name in front_half:
        config["overrides"].append({
            "pattern": layer_name,
            "wq": front_precision,
            "group_size": front_group_size,
            "comment": f"Front half expert (layer {layer_idx})"
        })
    
    # Assign back half experts
    for layer_idx, layer_name in back_half:
        config["overrides"].append({
            "pattern": layer_name,
            "wq": back_precision,
            "group_size": back_group_size,
            "comment": f"Back half expert (layer {layer_idx})"
        })
    
    return config, len(front_half), len(back_half)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Step 2.1: Uniform bitwidth allocation for experts"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model ID or local path"
    )
    parser.add_argument(
        "--front-precision",
        type=str,
        default="mxfp6",
        help="Precision for front half experts (e.g., mxfp4, mxfp6, mxfp8, int3, int4)"
    )
    parser.add_argument(
        "--back-precision",
        type=str,
        default="mxfp4",
        help="Precision for back half experts (e.g., mxfp4, mxfp6, mxfp8, int3, int4)"
    )
    parser.add_argument(
        "--front-group-size",
        type=int,
        default=128,
        help="Group size for front half experts"
    )
    parser.add_argument(
        "--back-group-size",
        type=int,
        default=128,
        help="Group size for back half experts"
    )
    parser.add_argument(
        "--default-precision",
        type=str,
        default="mxfp8",
        help="Default precision for non-expert layers"
    )
    parser.add_argument(
        "--default-group-size",
        type=int,
        default=128,
        help="Default group size for non-expert layers"
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
        default="configs/uniform_quant_config.json",
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
    
    # Load model (just to analyze structure)
    logger.info(f"Loading model to analyze structure: {args.model}")
    model = load_hf_causal_lm(
        model_id=args.model,
        device=args.device,
        dtype="fp32"
    )
    
    # Create quantization config
    logger.info("Creating uniform quantization configuration...")
    logger.info(f"  Front half precision: {args.front_precision} (group_size={args.front_group_size})")
    logger.info(f"  Back half precision: {args.back_precision} (group_size={args.back_group_size})")
    logger.info(f"  Default precision: {args.default_precision} (group_size={args.default_group_size})")
    logger.info(f"  Skip attention layers: {args.skip_attention}")
    logger.info(f"  Skip gate/router layers: {args.skip_gate}")
    
    config, n_front, n_back = create_uniform_quantization_config(
        model,
        front_precision=args.front_precision,
        back_precision=args.back_precision,
        front_group_size=args.front_group_size,
        back_group_size=args.back_group_size,
        skip_attention=args.skip_attention,
        skip_gate=args.skip_gate,
        default_precision=args.default_precision,
        default_group_size=args.default_group_size
    )
    
    # Save config
    logger.info(f"Saving config to: {args.output}")
    with open(args.output, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"✓ Uniform bitwidth allocation complete!")
    logger.info(f"  Front half experts: {n_front} → {args.front_precision}")
    logger.info(f"  Back half experts: {n_back} → {args.back_precision}")
    logger.info(f"  Total expert layers: {n_front + n_back}")
    logger.info(f"\nNext step: Use 3_gptq_quantize.py to quantize the model")


if __name__ == "__main__":
    main()

