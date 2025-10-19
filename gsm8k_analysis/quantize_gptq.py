"""
GPTQ quantization with configurable calibration data.

This script applies GPTQ quantization to OLMoE using different calibration datasets.
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
from data_utils import GSM8KCalibrationDataLoader


def create_gptq_plan_for_experts(model, bits: int = 4) -> dict:
    """
    Create GPTQ quantization plan for all expert layers.
    
    Args:
        model: The model to create plan for
        bits: Number of bits for quantization
        
    Returns:
        Dictionary mapping layer names to quantization schemes
    """
    plan = {}
    
    # Find all expert layers
    for name, module in model.named_modules():
        # Check if this is an expert layer
        if 'mlp.experts' in name and any(w in name for w in ['w1', 'w2', 'w3']):
            # Apply INT quantization with GPTQ
            plan[name] = {
                "wq": f"int{bits}",
                "aq": None,
                "group_size": 128,
                "extra": {}
            }
    
    return plan


def parse_args():
    parser = argparse.ArgumentParser(
        description="GPTQ quantization with configurable calibration data"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="allenai/OLMoE-1B-7B-0924",
        help="HuggingFace model ID or path"
    )
    parser.add_argument(
        "--calibration_data",
        type=str,
        default="wikitext2",
        choices=["wikitext2", "gsm8k", "c4"],
        help="Calibration dataset to use"
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
        "--bits",
        type=int,
        default=4,
        choices=[2, 3, 4, 8],
        help="Quantization bits"
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
        default=None,
        help="Output JSON file for results (auto-generated if not specified)"
    )
    parser.add_argument(
        "--save_plan",
        type=str,
        default=None,
        help="Save quantization plan to file (auto-generated if not specified)"
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
    
    # Auto-generate output paths if not specified
    if args.output is None:
        args.output = f"gsm8k_analysis/results/gptq_{args.calibration_data}_results.json"
    if args.save_plan is None:
        args.save_plan = f"gsm8k_analysis/configs/gptq_{args.calibration_data}_plan.json"
    
    print("=" * 60)
    print(f"GPTQ Quantization (Calibration: {args.calibration_data})")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Calibration data: {args.calibration_data}")
    print(f"Calibration samples: {args.num_calibration_samples}")
    print(f"Eval samples: {args.num_eval_samples}")
    print(f"Bits: {args.bits}")
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
        device_map="cpu",  # Load on CPU first, will move to device during GPTQ
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    
    # Create quantization plan
    print("\nCreating GPTQ quantization plan...")
    plan = create_gptq_plan_for_experts(model, bits=args.bits)
    print(f"Created plan for {len(plan)} layers")
    
    # Save plan
    print(f"Saving plan to: {args.save_plan}")
    with open(args.save_plan, 'w') as f:
        json.dump(plan, f, indent=2)
    
    # Load calibration data
    print(f"\nLoading {args.calibration_data} calibration data...")
    if args.calibration_data == "gsm8k":
        dataloader = GSM8KCalibrationDataLoader(
            nsamples=args.num_calibration_samples,
            seqlen=args.seqlen,
            tokenizer=tokenizer,
            split='train'
        )
    else:
        dataloader = CalibrationDataLoader(
            dataset_name=args.calibration_data,
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
    print("\nApplying GPTQ quantization...")
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
            'calibration_data': args.calibration_data,
            'num_calibration_samples': args.num_calibration_samples
        }, args.save_model)
    
    # Move model to device for evaluation
    model = model.to(args.device)
    
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
        limit=args.num_eval_samples
    )
    
    # Print results
    print("\n" + "=" * 60)
    print(f"GPTQ ({args.calibration_data}) Evaluation Results")
    print("=" * 60)
    print(make_table(results))
    
    # Save results
    print(f"\nSaving results to: {args.output}")
    results_with_metadata = {
        "results": results,
        "quantization_plan": args.save_plan,
        "calibration_data": args.calibration_data,
        "num_calibration_samples": args.num_calibration_samples,
        "num_quantized_layers": len(quantizers),
        "bits": args.bits,
        "gptq_config": {
            "percdamp": args.percdamp,
            "blocksize": args.blocksize,
            "actorder": args.actorder
        }
    }
    with open(args.output, 'w') as f:
        json.dump(results_with_metadata, f, indent=2)
    
    # Extract accuracy
    if 'results' in results and 'gsm8k' in results['results']:
        gsm8k_results = results['results']['gsm8k']
        if 'exact_match,strict-match' in gsm8k_results:
            accuracy = gsm8k_results['exact_match,strict-match']
            print(f"\nGSM8K Accuracy: {accuracy:.4f}")
    
    print(f"\n✓ GPTQ ({args.calibration_data}) quantization and evaluation complete!")


if __name__ == '__main__':
    main()

