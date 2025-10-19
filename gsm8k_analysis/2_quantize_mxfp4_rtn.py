#!/usr/bin/env python
"""
Step 2: Quantize All Experts to MXFP4 (RTN - No Calibration)

This script applies MXFP4 quantization using RTN (Round-to-Nearest) to all expert layers.
RTN is fast but doesn't use calibration data.

Usage:
    python gsm8k_analysis/2_quantize_mxfp4_rtn.py \\
        --model allenai/OLMoE-1B-7B-0924 \\
        --num_samples 256 \\
        --output gsm8k_analysis/results/mxfp4_rtn.json
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
    """Create MXFP4 quantization plan for all expert layers."""
    plan = {}
    
    for name, module in model.named_modules():
        if 'mlp.experts' in name and any(w in name for w in ['w1', 'w2', 'w3']):
            plan[name] = {
                "wq": "mxfp4",
                "aq": None,
                "group_size": 128,
                "extra": {"format": "e8m0"}
            }
    
    return plan


def parse_args():
    parser = argparse.ArgumentParser(
        description="Step 2: Quantize experts to MXFP4 (RTN)"
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
        default="gsm8k_analysis/results/mxfp4_rtn.json",
        help="Output JSON file for results"
    )
    parser.add_argument(
        "--save_plan",
        type=str,
        default="gsm8k_analysis/configs/mxfp4_rtn_plan.json",
        help="Save quantization plan to file"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 70)
    print("Step 2: MXFP4 Quantization (RTN) + GSM8K Evaluation")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Method: RTN (no calibration)")
    print(f"Samples: {args.num_samples}")
    print(f"Device: {args.device}")
    print("=" * 70)
    
    # Create output directories
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    plan_path = Path(args.save_plan)
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load model
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]
    
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map=args.device,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    
    # Create and save quantization plan
    print("\nCreating MXFP4 quantization plan...")
    plan = create_mxfp4_plan_for_experts(model)
    print(f"Created plan for {len(plan)} layers")
    
    with open(args.save_plan, 'w') as f:
        json.dump(plan, f, indent=2)
    print(f"Saved plan to: {args.save_plan}")
    
    # Apply quantization
    print("\nApplying MXFP4 quantization (RTN)...")
    replaced = apply_layer_wise_quantization(model, plan, args.dtype)
    print(f"Quantized {len(replaced)} modules")
    
    # Evaluate
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.batch_size, device=args.device)
    
    print("\nRunning GSM8K evaluation...")
    results = evaluator.simple_evaluate(model=lm, tasks=["gsm8k"], batch_size=args.batch_size, limit=args.num_samples)
    
    print("\n" + "=" * 70)
    print("MXFP4 (RTN) Evaluation Results")
    print("=" * 70)
    print(make_table(results))
    
    # Save results
    results_with_plan = {
        "results": results,
        "quantization_plan": args.save_plan,
        "num_quantized_layers": len(replaced),
        "quantization_format": "MXFP4"
    }
    with open(args.output, 'w') as f:
        json.dump(results_with_plan, f, indent=2)
    print(f"\nSaved results to: {args.output}")
    
    if 'results' in results and 'gsm8k' in results['results']:
        gsm8k_results = results['results']['gsm8k']
        if 'exact_match,strict-match' in gsm8k_results:
            accuracy = gsm8k_results['exact_match,strict-match']
            print(f"\nGSM8K Accuracy: {accuracy:.4f}")
    
    print("\n✓ Step 2 complete!")


if __name__ == '__main__':
    main()

