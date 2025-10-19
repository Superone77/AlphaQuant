#!/usr/bin/env python
"""
GSM8K Analysis Pipeline

This script runs the complete GSM8K analysis pipeline:
1. Evaluate baseline OLMoE on 256 GSM8K samples
2. Quantize all experts to MXFP4 (RTN - no calibration) and evaluate
3. GPTQ + MXFP4 quantization with wikitext2 calibration and evaluate
4. GPTQ + MXFP4 quantization with GSM8K calibration and evaluate
5. Compare and analyze results

Usage:
    python gsm8k_analysis/run_pipeline.py --model allenai/OLMoE-1B-7B-0924 --num_samples 256
"""

import sys
import os
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import subprocess
import json
from datetime import datetime


def run_command(cmd, description):
    """Run a command and handle errors."""
    print("\n" + "=" * 70)
    print(f"Running: {description}")
    print("=" * 70)
    print(f"Command: {' '.join(cmd)}")
    print()
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"\n❌ Error running {description}")
        print(f"Return code: {result.returncode}")
        return False
    
    print(f"\n✓ {description} completed successfully")
    return True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run GSM8K analysis pipeline"
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
        "--num_calibration_samples",
        type=int,
        default=128,
        help="Number of calibration samples for GPTQ"
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
        "--skip_baseline",
        action="store_true",
        help="Skip baseline evaluation"
    )
    parser.add_argument(
        "--skip_mxfp4",
        action="store_true",
        help="Skip MXFP4 (RTN) quantization"
    )
    parser.add_argument(
        "--skip_gptq_wikitext",
        action="store_true",
        help="Skip GPTQ + MXFP4 with wikitext2"
    )
    parser.add_argument(
        "--skip_gptq_gsm8k",
        action="store_true",
        help="Skip GPTQ + MXFP4 with GSM8K"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="gsm8k_analysis/results",
        help="Output directory for results"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("=" * 70)
    print("GSM8K Analysis Pipeline")
    print("=" * 70)
    print(f"Timestamp: {timestamp}")
    print(f"Model: {args.model}")
    print(f"Samples: {args.num_samples}")
    print(f"Calibration samples: {args.num_calibration_samples}")
    print(f"Device: {args.device}")
    print(f"Dtype: {args.dtype}")
    print("=" * 70)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Track all results
    all_results = {
        "timestamp": timestamp,
        "model": args.model,
        "num_samples": args.num_samples,
        "num_calibration_samples": args.num_calibration_samples,
        "experiments": {}
    }
    
    success = True
    
    # Step 1: Baseline evaluation
    if not args.skip_baseline:
        baseline_output = output_dir / f"baseline_{timestamp}.json"
        cmd = [
            "python", "gsm8k_analysis/eval_baseline.py",
            "--model", args.model,
            "--num_samples", str(args.num_samples),
            "--batch_size", str(args.batch_size),
            "--device", args.device,
            "--dtype", args.dtype,
            "--output", str(baseline_output)
        ]
        
        if run_command(cmd, "Step 1: Baseline Evaluation"):
            all_results["experiments"]["baseline"] = str(baseline_output)
        else:
            success = False
    
    # Step 2: MXFP4 (RTN) quantization
    if success and not args.skip_mxfp4:
        mxfp4_output = output_dir / f"mxfp4_rtn_{timestamp}.json"
        mxfp4_plan = output_dir.parent / "configs" / f"mxfp4_rtn_plan_{timestamp}.json"
        cmd = [
            "python", "gsm8k_analysis/quantize_mxfp4.py",
            "--model", args.model,
            "--num_samples", str(args.num_samples),
            "--batch_size", str(args.batch_size),
            "--device", args.device,
            "--dtype", args.dtype,
            "--output", str(mxfp4_output),
            "--save_plan", str(mxfp4_plan)
        ]
        
        if run_command(cmd, "Step 2: MXFP4 (RTN) Quantization"):
            all_results["experiments"]["mxfp4_rtn"] = str(mxfp4_output)
        else:
            success = False
    
    # Step 3: GPTQ + MXFP4 with wikitext2
    if success and not args.skip_gptq_wikitext:
        gptq_wikitext_output = output_dir / f"gptq_mxfp4_wikitext2_{timestamp}.json"
        gptq_wikitext_plan = output_dir.parent / "configs" / f"gptq_mxfp4_wikitext2_plan_{timestamp}.json"
        cmd = [
            "python", "gsm8k_analysis/quantize_gptq.py",
            "--model", args.model,
            "--calibration_data", "wikitext2",
            "--num_calibration_samples", str(args.num_calibration_samples),
            "--num_eval_samples", str(args.num_samples),
            "--batch_size", str(args.batch_size),
            "--device", args.device,
            "--dtype", args.dtype,
            "--output", str(gptq_wikitext_output),
            "--save_plan", str(gptq_wikitext_plan)
        ]
        
        if run_command(cmd, "Step 3: GPTQ + MXFP4 with WikiText2"):
            all_results["experiments"]["gptq_mxfp4_wikitext2"] = str(gptq_wikitext_output)
        else:
            success = False
    
    # Step 4: GPTQ + MXFP4 with GSM8K
    if success and not args.skip_gptq_gsm8k:
        gptq_gsm8k_output = output_dir / f"gptq_mxfp4_gsm8k_{timestamp}.json"
        gptq_gsm8k_plan = output_dir.parent / "configs" / f"gptq_mxfp4_gsm8k_plan_{timestamp}.json"
        cmd = [
            "python", "gsm8k_analysis/quantize_gptq.py",
            "--model", args.model,
            "--calibration_data", "gsm8k",
            "--num_calibration_samples", str(args.num_calibration_samples),
            "--num_eval_samples", str(args.num_samples),
            "--batch_size", str(args.batch_size),
            "--device", args.device,
            "--dtype", args.dtype,
            "--output", str(gptq_gsm8k_output),
            "--save_plan", str(gptq_gsm8k_plan)
        ]
        
        if run_command(cmd, "Step 4: GPTQ + MXFP4 with GSM8K"):
            all_results["experiments"]["gptq_mxfp4_gsm8k"] = str(gptq_gsm8k_output)
        else:
            success = False
    
    # Save pipeline summary
    summary_path = output_dir / f"pipeline_summary_{timestamp}.json"
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Step 5: Analyze results
    if success:
        print("\n" + "=" * 70)
        print("Running: Step 5: Results Analysis")
        print("=" * 70)
        
        cmd = [
            "python", "gsm8k_analysis/analyze_results.py",
            "--summary", str(summary_path)
        ]
        
        run_command(cmd, "Step 5: Results Analysis")
    
    # Print final summary
    print("\n" + "=" * 70)
    print("Pipeline Complete!")
    print("=" * 70)
    print(f"Summary saved to: {summary_path}")
    
    if success:
        print("\n✓ All steps completed successfully!")
    else:
        print("\n❌ Some steps failed. Check the logs above.")
    
    return 0 if success else 1


if __name__ == '__main__':
    exit(main())

