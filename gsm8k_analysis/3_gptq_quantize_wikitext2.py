#!/usr/bin/env python
"""
Step 3: GPTQ + MXFP4 Quantization with WikiText2 Calibration

This script applies GPTQ quantization to expert layers using WikiText2 as calibration data.
Uses MoE-optimized GPTQ with routing-weighted Hessian computation.

Usage:
    python gsm8k_analysis/3_gptq_quantize_wikitext2.py \\
        --model allenai/OLMoE-1B-7B-0924 \\
        --num_calibration_samples 128 \\
        --num_eval_samples 256 \\
        --output gsm8k_analysis/results/gptq_mxfp4_wikitext2.json
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

from alphaquant.gptq.quantize import gptq_quantize_model
from alphaquant.gptq.gptq import GPTQConfig
from alphaquant.gptq.data_utils import CalibrationDataLoader
from alphaquant.utils.eval_utils import make_table


def create_gptq_mxfp4_plan_for_experts(model) -> dict:
    """Create GPTQ + MXFP4 quantization plan for all expert layers."""
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
        description="Step 3: GPTQ + MXFP4 with WikiText2 calibration"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="allenai/OLMoE-1B-7B-0924",
        help="HuggingFace model ID or path"
    )
    parser.add_argument(
        "--num_calibration_samples",
        type=int,
        default=128,
        help="Number of calibration samples"
    )
    parser.add_argument(
        "--num_eval_samples",
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
        "--seqlen",
        type=int,
        default=2048,
        help="Sequence length for calibration"
    )
    parser.add_argument(
        "--percdamp",
        type=float,
        default=0.01,
        help="GPTQ percdamp parameter"
    )
    parser.add_argument(
        "--blocksize",
        type=int,
        default=128,
        help="GPTQ blocksize parameter"
    )
    parser.add_argument(
        "--actorder",
        action="store_true",
        help="Use activation order in GPTQ"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="gsm8k_analysis/results/gptq_mxfp4_wikitext2.json",
        help="Output JSON file for results"
    )
    parser.add_argument(
        "--save_plan",
        type=str,
        default="gsm8k_analysis/configs/gptq_mxfp4_wikitext2_plan.json",
        help="Save quantization plan to file"
    )
    parser.add_argument(
        "--save_model",
        type=str,
        default=None,
        help="Save quantized model checkpoint (optional)"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 70)
    print("Step 3: GPTQ + MXFP4 Quantization (WikiText2 Calibration)")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Calibration data: WikiText2")
    print(f"Calibration samples: {args.num_calibration_samples}")
    print(f"Eval samples: {args.num_eval_samples}")
    print(f"Quantization format: MXFP4")
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
        device_map="cpu",  # Load on CPU first
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    
    # Create and save quantization plan
    print("\nCreating GPTQ + MXFP4 quantization plan...")
    plan = create_gptq_mxfp4_plan_for_experts(model)
    print(f"Created plan for {len(plan)} layers")
    
    with open(args.save_plan, 'w') as f:
        json.dump(plan, f, indent=2)
    print(f"Saved plan to: {args.save_plan}")
    
    # Load calibration data
    print(f"\nLoading WikiText2 calibration data...")
    dataloader = CalibrationDataLoader(
        dataset_name="wikitext2",
        nsamples=args.num_calibration_samples,
        seqlen=args.seqlen,
        tokenizer=tokenizer
    )
    
    # Create GPTQ config
    gptq_config = GPTQConfig(
        percdamp=args.percdamp,
        blocksize=args.blocksize,
        actorder=args.actorder,
        static_groups=False,
        use_hadamard=False
    )
    
    # Apply GPTQ quantization
    print("\nApplying GPTQ + MXFP4 quantization...")
    print("This may take a while...")
    quantizers = gptq_quantize_model(
        model=model,
        dataloader=dataloader,
        layer_config=plan,
        device=args.device,
        gptq_config=gptq_config,
        model_type='auto',
        dtype=args.dtype
    )
    print(f"Quantized {len(quantizers)} modules")
    
    # Save model checkpoint if requested
    if args.save_model:
        print(f"\nSaving quantized model to: {args.save_model}")
        save_path = Path(args.save_model)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'quantization_plan': plan,
            'gptq_config': {
                'percdamp': args.percdamp,
                'blocksize': args.blocksize,
                'actorder': args.actorder
            },
            'calibration_data': 'wikitext2',
            'num_calibration_samples': args.num_calibration_samples,
            'quantization_format': 'MXFP4'
        }, args.save_model)
    
    # Move model to device for evaluation
    model = model.to(args.device)
    
    # Evaluate
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.batch_size, device=args.device)
    
    print("\nRunning GSM8K evaluation...")
    results = evaluator.simple_evaluate(
        model=lm,
        tasks=["gsm8k"],
        batch_size=args.batch_size,
        limit=args.num_eval_samples
    )
    
    print("\n" + "=" * 70)
    print("GPTQ + MXFP4 (WikiText2) Evaluation Results")
    print("=" * 70)
    print(make_table(results))
    
    # Save results
    results_with_metadata = {
        "results": results,
        "quantization_plan": args.save_plan,
        "calibration_data": "wikitext2",
        "num_calibration_samples": args.num_calibration_samples,
        "num_quantized_layers": len(quantizers),
        "quantization_format": "MXFP4",
        "gptq_config": {
            "percdamp": args.percdamp,
            "blocksize": args.blocksize,
            "actorder": args.actorder
        }
    }
    with open(args.output, 'w') as f:
        json.dump(results_with_metadata, f, indent=2)
    print(f"\nSaved results to: {args.output}")
    
    if 'results' in results and 'gsm8k' in results['results']:
        gsm8k_results = results['results']['gsm8k']
        if 'exact_match,strict-match' in gsm8k_results:
            accuracy = gsm8k_results['exact_match,strict-match']
            print(f"\nGSM8K Accuracy: {accuracy:.4f}")
    
    print("\n✓ Step 3 complete!")


if __name__ == '__main__':
    main()

