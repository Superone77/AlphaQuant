#!/usr/bin/env python
from __future__ import annotations
import argparse
import json
import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
import torch
from alphaquant.utils.replacement import load_quantization_plan, create_quantizer_from_scheme, apply_layer_wise_quantization
from alphaquant.utils.eval_utils import tasks_evaluate



def main():
    parser = argparse.ArgumentParser(description="Evaluate OLMoE with layer-wise quantization from plans")
    parser.add_argument('--model', required=True, help='Path to OLMoE model or HF model ID')
    parser.add_argument('--plan', required=True, help='Path to quantization plan JSON file')
    parser.add_argument('--tasks', default='gsm8k', help='Comma-separated list of tasks (default: gsm8k)')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for evaluation')
    parser.add_argument('--device', default='cuda:0', help='Device to use (cuda/cpu)')
    parser.add_argument('--dtype', default='bfloat16', help='Model dtype (bfloat16/float16/float32)')
    parser.add_argument('--dry_run', action='store_true', help='Only show quantization plan without applying')
    
    args = parser.parse_args()
    
    # Load quantization plan
    print(f"Loading quantization plan from: {args.plan}")
    plan = load_quantization_plan(args.plan)
    print(f"Loaded plan with {len(plan)} modules")
    
    if args.dry_run:
        print("Dry run mode - showing first 10 modules in plan:")
        for i, (name, scheme) in enumerate(list(plan.items())[:10]):
            print(f"  {name}: {scheme}")
        if len(plan) > 10:
            print(f"  ... and {len(plan) - 10} more modules")
        return
    
    # Build HFLM
    print(f"Loading model: {args.model}")
    lm = HFLM(pretrained=args.model, device=args.device, dtype=args.dtype, batch_size=args.batch_size)
    model = lm.model
    
    # Apply quantization
    print("Applying layer-wise quantization...")
    replaced = apply_layer_wise_quantization(model, plan, args.dtype)
    print(f"Quantized {len(replaced)} modules")
    
    # Run evaluation
    args.tasks = args.tasks.split(",")
    print(f"Running evaluation on tasks: {args.tasks}")
    tasks_evaluate(model = lm, tasks = args.tasks, batch_size = args.batch_size,device=args.device)
    # results = evaluator.simple_evaluate(
    #     model=lm,
    #     tasks=args.tasks.split(','),
    #     batch_size=args.batch_size,
    # )
    
    # print("\n" + "="*50)
    # print("EVALUATION RESULTS")
    # print("="*50)
    # print(results)
    
    # # Save results
    # output_file = f"olmoe_quantized_{args.tasks.replace(',', '_')}_results.json"
    # with open(output_file, 'w') as f:
    #     json.dump(results, f, indent=2)
    # print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main() 