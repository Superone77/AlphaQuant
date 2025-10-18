#!/usr/bin/env python
"""
Step 4: Evaluate Quantized Model

This script evaluates the quantized model from step 3 on various benchmarks.

Usage:
    python 4_evaluate_model.py \\
        --model allenai/OLMoE-1B-7B-0924 \\
        --checkpoint quantized_model.pt \\
        --tasks hellaswag,arc_easy,winogrande \\
        --output results/eval_results.json

Output:
    - JSON file with evaluation results
    - Can be analyzed in step 5
"""

import sys
import os

ROOT = os.path.abspath(os.path.dirname(__file__))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json
import torch
from pathlib import Path

from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from alphaquant.utils.replacement import apply_layer_wise_quantization


def parse_args():
    parser = argparse.ArgumentParser(
        description="Step 4: Evaluate quantized model"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Original HuggingFace model ID or path"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Quantized model checkpoint from step 3 (optional)"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Quantization config JSON (if not using checkpoint)"
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="hellaswag,arc_easy,winogrande",
        help="Comma-separated list of evaluation tasks"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda/cpu)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/eval_results.json",
        help="Output JSON file for results"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("="*60)
    print("Step 4: Evaluate Quantized Model")
    print("="*60)
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"Loading model: {args.model}")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    
    # Load quantization if provided
    if args.checkpoint:
        print(f"Loading quantized checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        config = checkpoint.get('config', {})
        plan = checkpoint.get('plan', {})
    elif args.config:
        print(f"Loading quantization config: {args.config}")
        with open(args.config, 'r') as f:
            config = json.load(f)
        # Apply quantization
        plan = apply_layer_wise_quantization(model, config)
    else:
        print("No quantization applied (evaluating original model)")
        plan = {}
    
    # Parse tasks
    task_list = [t.strip() for t in args.tasks.split(',')]
    print(f"Evaluating on tasks: {task_list}")
    
    # Create HF language model wrapper
    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        device=args.device
    )
    
    # Run evaluation
    print("\nRunning evaluation...")
    results = evaluator.simple_evaluate(
        model=lm,
        tasks=task_list,
        batch_size=args.batch_size
    )
    
    # Save results
    print(f"\nSaving results to: {args.output}")
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("Evaluation Results:")
    print("="*60)
    for task in task_list:
        if task in results['results']:
            task_results = results['results'][task]
            print(f"\n{task}:")
            for metric, value in task_results.items():
                if isinstance(value, (int, float)):
                    print(f"  {metric}: {value:.4f}")
    
    print("\n✓ Evaluation complete!")
    print(f"\nNext step: Use 5_analyze_results.py to analyze the results")


if __name__ == "__main__":
    main()

