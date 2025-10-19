#!/usr/bin/env python
"""
Example usage of GSM8K analysis tools.

This script demonstrates how to use the individual components
of the GSM8K analysis pipeline programmatically.
"""

import sys
import os
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def example_load_gsm8k_data():
    """Example: Load GSM8K calibration data."""
    from transformers import AutoTokenizer
    from gsm8k_analysis.data_utils import GSM8KCalibrationDataLoader
    
    print("=" * 60)
    print("Example 1: Loading GSM8K Calibration Data")
    print("=" * 60)
    
    # Load tokenizer
    model_name = "allenai/OLMoE-1B-7B-0924"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Create data loader
    dataloader = GSM8KCalibrationDataLoader(
        nsamples=10,
        seqlen=512,
        tokenizer=tokenizer,
        split='train'
    )
    
    # Iterate over samples
    print(f"\nLoading {len(dataloader)} samples...")
    for i, batch in enumerate(dataloader):
        print(f"Sample {i+1}: shape={batch.shape}")
        if i >= 2:  # Just show first few
            break
    
    print("\n✓ Data loading successful!\n")


def example_evaluate_model():
    """Example: Evaluate model on GSM8K."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM
    
    print("=" * 60)
    print("Example 2: Evaluate Model on GSM8K")
    print("=" * 60)
    
    model_name = "allenai/OLMoE-1B-7B-0924"
    
    print(f"\nLoading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Create language model wrapper
    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=1,
        device="cuda"
    )
    
    # Evaluate on small subset
    print("\nEvaluating on 10 GSM8K samples...")
    results = evaluator.simple_evaluate(
        model=lm,
        tasks=["gsm8k"],
        batch_size=1,
        limit=10
    )
    
    # Extract accuracy
    if 'results' in results and 'gsm8k' in results['results']:
        gsm8k_results = results['results']['gsm8k']
        print(f"\nResults: {gsm8k_results}")
    
    print("\n✓ Evaluation complete!\n")


def example_create_quantization_plan():
    """Example: Create quantization plan for experts."""
    import torch
    from transformers import AutoModelForCausalLM
    import json
    
    print("=" * 60)
    print("Example 3: Create Quantization Plan")
    print("=" * 60)
    
    model_name = "allenai/OLMoE-1B-7B-0924"
    
    print(f"\nLoading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cpu",  # Just for inspection
        trust_remote_code=True
    )
    
    # Find expert layers
    print("\nFinding expert layers...")
    plan = {}
    
    for name, module in model.named_modules():
        if 'mlp.experts' in name and any(w in name for w in ['w1', 'w2', 'w3']):
            plan[name] = {
                "wq": "mxfp4",
                "aq": None,
                "group_size": 128,
                "extra": {"format": "e8m0"}
            }
    
    print(f"\nCreated plan for {len(plan)} layers")
    print("\nExample entries:")
    for i, (name, scheme) in enumerate(list(plan.items())[:3]):
        print(f"  {name}: {scheme}")
    
    # Save plan
    output_file = "gsm8k_analysis/configs/example_plan.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(plan, f, indent=2)
    
    print(f"\n✓ Plan saved to: {output_file}\n")


def example_analyze_results():
    """Example: Analyze results from JSON files."""
    import json
    import pandas as pd
    
    print("=" * 60)
    print("Example 4: Analyze Results")
    print("=" * 60)
    
    # Create example results
    example_results = {
        "baseline": {
            "results": {
                "gsm8k": {
                    "exact_match,strict-match": 0.6523,
                    "exact_match,strict-match_stderr": 0.0142
                }
            }
        },
        "mxfp4": {
            "results": {
                "gsm8k": {
                    "exact_match,strict-match": 0.5892,
                    "exact_match,strict-match_stderr": 0.0148
                }
            },
            "num_quantized_layers": 192
        },
        "gptq_gsm8k": {
            "results": {
                "gsm8k": {
                    "exact_match,strict-match": 0.6341,
                    "exact_match,strict-match_stderr": 0.0145
                }
            },
            "calibration_data": "gsm8k",
            "num_quantized_layers": 192,
            "bits": 4
        }
    }
    
    # Create comparison table
    comparison = []
    for name, data in example_results.items():
        gsm8k_res = data['results']['gsm8k']
        comparison.append({
            'Experiment': name,
            'Accuracy': gsm8k_res['exact_match,strict-match'],
            'Stderr': gsm8k_res['exact_match,strict-match_stderr'],
            'Calibration': data.get('calibration_data', 'N/A'),
            'Quantized Layers': data.get('num_quantized_layers', 'N/A'),
            'Bits': data.get('bits', 'N/A')
        })
    
    df = pd.DataFrame(comparison)
    df = df.sort_values('Accuracy', ascending=False)
    
    print("\nComparison Table:")
    print(df.to_string(index=False))
    
    # Calculate accuracy drops
    baseline_acc = df[df['Experiment'] == 'baseline']['Accuracy'].values[0]
    
    print("\nAccuracy Drop from Baseline:")
    for _, row in df.iterrows():
        if row['Experiment'] != 'baseline':
            acc = row['Accuracy']
            drop = baseline_acc - acc
            drop_pct = (drop / baseline_acc) * 100
            print(f"  {row['Experiment']:15s}: {drop:+.4f} ({drop_pct:+.2f}%)")
    
    print("\n✓ Analysis complete!\n")


def main():
    """Run all examples."""
    import argparse
    
    parser = argparse.ArgumentParser(description="GSM8K Analysis Examples")
    parser.add_argument(
        "--example",
        type=int,
        choices=[1, 2, 3, 4],
        help="Run specific example (1-4), or all if not specified"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Skip examples that require model loading (1, 2, 3)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("GSM8K Analysis - Example Usage")
    print("=" * 60 + "\n")
    
    examples = {
        1: ("Load GSM8K Data", example_load_gsm8k_data, False),
        2: ("Evaluate Model", example_evaluate_model, False),
        3: ("Create Quantization Plan", example_create_quantization_plan, False),
        4: ("Analyze Results", example_analyze_results, True)
    }
    
    if args.example:
        # Run specific example
        name, func, is_quick = examples[args.example]
        if args.quick and not is_quick:
            print(f"Skipping Example {args.example} (requires model loading)")
        else:
            func()
    else:
        # Run all examples
        for idx, (name, func, is_quick) in examples.items():
            if args.quick and not is_quick:
                print(f"Skipping Example {idx}: {name} (requires model loading)\n")
                continue
            
            try:
                func()
            except Exception as e:
                print(f"\n❌ Example {idx} failed: {e}\n")
                if not args.quick:
                    import traceback
                    traceback.print_exc()
    
    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()

