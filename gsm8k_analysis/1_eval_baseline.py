#!/usr/bin/env python
"""
Step 1: Evaluate Baseline OLMoE on GSM8K

This script evaluates the original (non-quantized) OLMoE model on GSM8K samples.

Usage:
    python gsm8k_analysis/1_eval_baseline.py \\
        --model allenai/OLMoE-1B-7B-0924 \\
        --num_samples 256 \\
        --output gsm8k_analysis/results/baseline.json
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

from alphaquant.utils.eval_utils import make_table
from gsm8k_analysis.eval_utils import save_samples_with_reasoning


def parse_args():
    parser = argparse.ArgumentParser(
        description="Step 1: Evaluate baseline OLMoE on GSM8K"
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
        default="gsm8k_analysis/results/baseline.json",
        help="Output JSON file for results"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 70)
    print("Step 1: GSM8K Baseline Evaluation")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Samples: {args.num_samples}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {args.device}")
    print(f"Dtype: {args.dtype}")
    print("=" * 70)
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
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
        limit=args.num_samples,
        log_samples=True  # Enable detailed sample logging
    )
    
    # Print results
    print("\n" + "=" * 70)
    print("Baseline Evaluation Results")
    print("=" * 70)
    print(make_table(results))
    
    # Save aggregate results
    print(f"\nSaving aggregate results to: {args.output}")
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save detailed sample-level results with reasoning
    detailed_output = args.output.replace('.json', '_samples.json')
    save_samples_with_reasoning(results, detailed_output, task_name="gsm8k")
    
    # Extract accuracy
    if 'results' in results and 'gsm8k' in results['results']:
        gsm8k_results = results['results']['gsm8k']
        if 'exact_match,strict-match' in gsm8k_results:
            accuracy = gsm8k_results['exact_match,strict-match']
            print(f"\nGSM8K Accuracy: {accuracy:.4f}")
    
    print("\n✓ Step 1 complete!")
    print(f"  - Aggregate results: {args.output}")
    print(f"  - Detailed samples: {detailed_output}")


if __name__ == '__main__':
    main()

