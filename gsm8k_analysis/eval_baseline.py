"""
Evaluate baseline OLMoE model on GSM8K.

This script evaluates the original (non-quantized) OLMoE model on a subset of GSM8K samples.
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate baseline OLMoE on GSM8K"
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
        default="gsm8k_analysis/results/baseline_results.json",
        help="Output JSON file for results"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("GSM8K Baseline Evaluation")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Samples: {args.num_samples}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {args.device}")
    print(f"Dtype: {args.dtype}")
    print("=" * 60)
    
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
    
    # Configure to use only the specified number of samples
    eval_config = {
        "limit": args.num_samples
    }
    
    results = evaluator.simple_evaluate(
        model=lm,
        tasks=["gsm8k"],
        batch_size=args.batch_size,
        limit=args.num_samples
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("Baseline Evaluation Results")
    print("=" * 60)
    print(make_table(results))
    
    # Save results
    print(f"\nSaving results to: {args.output}")
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Extract accuracy
    if 'results' in results and 'gsm8k' in results['results']:
        gsm8k_results = results['results']['gsm8k']
        if 'exact_match,strict-match' in gsm8k_results:
            accuracy = gsm8k_results['exact_match,strict-match']
            print(f"\nGSM8K Accuracy: {accuracy:.4f}")
    
    print("\n✓ Baseline evaluation complete!")


if __name__ == '__main__':
    main()

