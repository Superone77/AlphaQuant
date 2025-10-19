"""
Quantize all experts to MXFP4 and evaluate on GSM8K.

This script applies MXFP4 quantization to all expert layers in OLMoE using RTN (no calibration).
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
import torch
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from transformers import AutoModelForCausalLM, AutoTokenizer

from alphaquant.utils.replacement import apply_layer_wise_quantization
from alphaquant.utils.eval_utils import make_table


def create_mxfp4_plan_for_experts(model) -> dict:
    """
    Create quantization plan to quantize all expert layers to MXFP4.
    
    Args:
        model: The model to create plan for
        
    Returns:
        Dictionary mapping layer names to quantization schemes
    """
    plan = {}
    
    # Find all expert layers
    # For OLMoE, experts are in: model.layers.{i}.mlp.experts.{j}.{w1,w2,w3}
    for name, module in model.named_modules():
        # Check if this is an expert layer
        if 'mlp.experts' in name and any(w in name for w in ['w1', 'w2', 'w3']):
            # Apply MXFP4 quantization (RTN - no calibration)
            plan[name] = {
                "wq": "mxfp4",
                "aq": None,  # No activation quantization
                "group_size": 128,
                "extra": {
                    "format": "e8m0"
                }
            }
    
    return plan


def parse_args():
    parser = argparse.ArgumentParser(
        description="Quantize OLMoE experts to MXFP4 and evaluate"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="allenai/OLMoE-1B-7B-0924",
        help="HuggingFace model ID or path"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=256,
        help="Number of GSM8K samples to evaluate"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda/cpu)"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Model dtype"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="gsm8k_analysis/results/mxfp4_results.json",
        help="Output JSON file for results"
    )
    parser.add_argument(
        "--save_plan",
        type=str,
        default="gsm8k_analysis/configs/mxfp4_plan.json",
        help="Save quantization plan to file"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("MXFP4 Quantization (RTN) + GSM8K Evaluation")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Samples: {args.num_samples}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {args.device}")
    print(f"Dtype: {args.dtype}")
    print("=" * 60)
    
    # Create output directories
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    plan_path = Path(args.save_plan)
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Determine dtype
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32
    }
    dtype = dtype_map[args.dtype]
    
    # Load model
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map=args.device,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    
    # Create quantization plan
    print("\nCreating MXFP4 quantization plan...")
    plan = create_mxfp4_plan_for_experts(model)
    print(f"Created plan for {len(plan)} layers")
    
    # Save plan
    print(f"Saving plan to: {args.save_plan}")
    with open(args.save_plan, 'w') as f:
        json.dump(plan, f, indent=2)
    
    # Apply quantization (RTN - Round to Nearest, no calibration needed)
    print("\nApplying MXFP4 quantization (RTN)...")
    replaced = apply_layer_wise_quantization(model, plan, args.dtype)
    print(f"Quantized {len(replaced)} modules")
    
    # Create HF language model wrapper
    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        device=args.device
    )
    
    # Run evaluation on GSM8K
    print("\nRunning GSM8K evaluation...")
    results = evaluator.simple_evaluate(
        model=lm,
        tasks=["gsm8k"],
        batch_size=args.batch_size,
        limit=args.num_samples
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("MXFP4 (RTN) Evaluation Results")
    print("=" * 60)
    print(make_table(results))
    
    # Save results
    print(f"\nSaving results to: {args.output}")
    results_with_plan = {
        "results": results,
        "quantization_plan": args.save_plan,
        "num_quantized_layers": len(replaced),
        "quantization_format": "MXFP4"
    }
    with open(args.output, 'w') as f:
        json.dump(results_with_plan, f, indent=2)
    
    # Extract accuracy
    if 'results' in results and 'gsm8k' in results['results']:
        gsm8k_results = results['results']['gsm8k']
        if 'exact_match,strict-match' in gsm8k_results:
            accuracy = gsm8k_results['exact_match,strict-match']
            print(f"\nGSM8K Accuracy: {accuracy:.4f}")
    
    print("\n✓ MXFP4 (RTN) quantization and evaluation complete!")


if __name__ == '__main__':
    main()
