#!/usr/bin/env python
"""
Analyze sample-level logs from GSM8K evaluations.

This script provides utilities to parse and analyze the detailed sample logs.

Usage:
    python gsm8k_analysis/analyze_sample_logs.py --log gsm8k_analysis/results/baseline_samples.log
"""

import sys
import os
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import re
from typing import List, Dict, Any


def parse_log_file(log_path: str) -> List[Dict[str, Any]]:
    """
    Parse a sample log file and extract structured data.
    
    Args:
        log_path: Path to the log file
        
    Returns:
        List of sample dictionaries
    """
    samples = []
    
    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by sample blocks
    sample_blocks = re.split(r'={100}\nSample #(\d+)\n={100}', content)
    
    # Process pairs (sample_number, content)
    for i in range(1, len(sample_blocks), 2):
        if i + 1 >= len(sample_blocks):
            break
        
        sample_num = int(sample_blocks[i])
        block = sample_blocks[i + 1]
        
        sample = {'index': sample_num}
        
        # Extract question
        question_match = re.search(r'Question:\n(.*?)\n\nGold Answer:', block, re.DOTALL)
        if question_match:
            sample['question'] = question_match.group(1).strip()
        
        # Extract gold answer
        gold_match = re.search(r'Gold Answer:\n(.*?)\n\nModel Reasoning:', block, re.DOTALL)
        if gold_match:
            sample['gold_answer'] = gold_match.group(1).strip()
        
        # Extract model reasoning
        reasoning_match = re.search(r'Model Reasoning:\n(.*?)\n\nModel Final Answer:', block, re.DOTALL)
        if reasoning_match:
            sample['model_reasoning'] = reasoning_match.group(1).strip()
        
        # Extract model answer
        answer_match = re.search(r'Model Final Answer:\n(.*?)\n\nCorrect:', block, re.DOTALL)
        if answer_match:
            sample['model_answer'] = answer_match.group(1).strip()
        
        # Extract correctness
        correct_match = re.search(r'Correct: (✓ YES|✗ NO)', block)
        if correct_match:
            sample['correct'] = correct_match.group(1) == '✓ YES'
        
        # Extract full output
        output_match = re.search(r'Full Model Output:\n-{50}\n(.*?)\n-{50}', block, re.DOTALL)
        if output_match:
            sample['full_output'] = output_match.group(1).strip()
        
        samples.append(sample)
    
    return samples


def print_sample_stats(samples: List[Dict[str, Any]], experiment_name: str = ""):
    """Print statistics about samples."""
    total = len(samples)
    correct = sum(1 for s in samples if s.get('correct', False))
    incorrect = total - correct
    accuracy = correct / total if total > 0 else 0
    
    print(f"\n{'='*70}")
    print(f"Statistics{f' - {experiment_name}' if experiment_name else ''}")
    print(f"{'='*70}")
    print(f"Total samples: {total}")
    print(f"Correct: {correct}")
    print(f"Incorrect: {incorrect}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy:.2%})")


def print_incorrect_samples(samples: List[Dict[str, Any]], max_show: int = 5):
    """Print incorrect samples."""
    incorrect = [s for s in samples if not s.get('correct', False)]
    
    print(f"\n{'='*70}")
    print(f"Incorrect Samples (showing first {min(max_show, len(incorrect))} of {len(incorrect)})")
    print(f"{'='*70}")
    
    for i, sample in enumerate(incorrect[:max_show]):
        print(f"\n{'-'*70}")
        print(f"Sample #{sample['index']}")
        print(f"{'-'*70}")
        print(f"Question: {sample.get('question', '')[:150]}...")
        print(f"\nGold Answer Extract: {sample.get('gold_answer', '')[:100]}...")
        print(f"\nModel Answer: {sample.get('model_answer', '')}")
        print(f"Model Reasoning: {sample.get('model_reasoning', '')[:200]}...")


def compare_experiments(log_paths: Dict[str, str]):
    """Compare multiple experiments."""
    all_samples = {}
    
    print(f"\n{'='*70}")
    print("Loading experiments...")
    print(f"{'='*70}")
    
    for name, path in log_paths.items():
        if not Path(path).exists():
            print(f"⚠ Skipping {name}: file not found")
            continue
        
        samples = parse_log_file(path)
        all_samples[name] = samples
        print(f"✓ Loaded {name}: {len(samples)} samples")
    
    if not all_samples:
        print("No valid log files found!")
        return
    
    # Print comparison
    print(f"\n{'='*70}")
    print("Accuracy Comparison")
    print(f"{'='*70}")
    
    for name, samples in all_samples.items():
        correct = sum(1 for s in samples if s.get('correct', False))
        total = len(samples)
        accuracy = correct / total if total > 0 else 0
        print(f"{name:30s}: {correct:3d}/{total:3d} = {accuracy:.4f} ({accuracy:.2%})")
    
    # Find baseline
    if 'baseline' not in all_samples:
        return
    
    baseline_samples = all_samples['baseline']
    
    # Compare with baseline
    print(f"\n{'='*70}")
    print("Accuracy Drop from Baseline")
    print(f"{'='*70}")
    
    baseline_acc = sum(1 for s in baseline_samples if s.get('correct', False)) / len(baseline_samples)
    
    for name, samples in all_samples.items():
        if name == 'baseline':
            continue
        
        acc = sum(1 for s in samples if s.get('correct', False)) / len(samples)
        drop = baseline_acc - acc
        drop_pct = (drop / baseline_acc) * 100 if baseline_acc > 0 else 0
        print(f"{name:30s}: {drop:+.4f} ({drop_pct:+.2f}%)")
    
    # Find degraded samples
    print(f"\n{'='*70}")
    print("Degradation Analysis")
    print(f"{'='*70}")
    
    for name, samples in all_samples.items():
        if name == 'baseline' or len(samples) != len(baseline_samples):
            continue
        
        degraded_count = 0
        for i in range(len(baseline_samples)):
            baseline_correct = baseline_samples[i].get('correct', False)
            quant_correct = samples[i].get('correct', False)
            
            if baseline_correct and not quant_correct:
                degraded_count += 1
        
        print(f"{name:30s}: {degraded_count} samples degraded from baseline")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze GSM8K sample logs"
    )
    
    parser.add_argument(
        "--log",
        type=str,
        help="Path to a single log file to analyze"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare all experiments in results directory"
    )
    parser.add_argument(
        "--show_incorrect",
        type=int,
        default=5,
        help="Number of incorrect samples to show (default: 5)"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 70)
    print("GSM8K Sample Log Analyzer")
    print("=" * 70)
    
    if args.compare:
        # Compare all experiments
        results_dir = Path("gsm8k_analysis/results")
        log_paths = {
            'baseline': str(results_dir / "baseline_samples.log"),
            'mxfp4_rtn': str(results_dir / "mxfp4_rtn_samples.log"),
            'gptq_wikitext2': str(results_dir / "gptq_mxfp4_wikitext2_samples.log"),
            'gptq_gsm8k': str(results_dir / "gptq_mxfp4_gsm8k_samples.log")
        }
        
        compare_experiments(log_paths)
    
    elif args.log:
        # Analyze single log file
        print(f"\nAnalyzing: {args.log}")
        
        if not Path(args.log).exists():
            print(f"Error: File not found: {args.log}")
            return 1
        
        samples = parse_log_file(args.log)
        
        experiment_name = Path(args.log).stem.replace('_samples', '')
        print_sample_stats(samples, experiment_name)
        print_incorrect_samples(samples, max_show=args.show_incorrect)
    
    else:
        print("\nPlease specify either --log <file> or --compare")
        print("\nExamples:")
        print("  python gsm8k_analysis/analyze_sample_logs.py --log gsm8k_analysis/results/baseline_samples.log")
        print("  python gsm8k_analysis/analyze_sample_logs.py --compare")
        return 1
    
    print("\n" + "=" * 70)
    print("✓ Analysis complete!")
    print("=" * 70)
    
    return 0


if __name__ == '__main__':
    exit(main())

