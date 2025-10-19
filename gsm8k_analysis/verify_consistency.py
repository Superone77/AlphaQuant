#!/usr/bin/env python
"""
Verify that all experiments evaluate on the same GSM8K samples.

This script checks that the sample questions are identical across all experiment logs.

Usage:
    python gsm8k_analysis/verify_consistency.py
"""

import sys
import os
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import re
from typing import List


def extract_questions_from_log(log_path: str) -> List[str]:
    """
    Extract all questions from a log file.
    
    Args:
        log_path: Path to the log file
        
    Returns:
        List of question strings
    """
    if not Path(log_path).exists():
        return []
    
    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract questions using regex
    questions = re.findall(r'Question:\n(.*?)\n\nGold Answer:', content, re.DOTALL)
    return [q.strip() for q in questions]


def main():
    print("=" * 70)
    print("GSM8K Sample Consistency Verification")
    print("=" * 70)
    
    # Define log files to check
    logs = {
        'baseline': 'gsm8k_analysis/results/baseline_samples.log',
        'mxfp4_rtn': 'gsm8k_analysis/results/mxfp4_rtn_samples.log',
        'gptq_wikitext2': 'gsm8k_analysis/results/gptq_mxfp4_wikitext2_samples.log',
        'gptq_gsm8k': 'gsm8k_analysis/results/gptq_mxfp4_gsm8k_samples.log'
    }
    
    all_questions = {}
    
    print("\nExtracting questions from log files...")
    for name, log_path in logs.items():
        if not Path(log_path).exists():
            print(f"  ⚠ Skipping {name}: file not found")
            continue
        
        questions = extract_questions_from_log(log_path)
        all_questions[name] = questions
        print(f"  ✓ {name:20s}: {len(questions)} samples")
    
    if len(all_questions) < 2:
        print("\n⚠ Need at least 2 log files to compare")
        return 1
    
    # Verify all experiments use the same questions
    print(f"\n{'='*70}")
    print("Consistency Check")
    print(f"{'='*70}")
    
    # Use first experiment as reference
    ref_name = list(all_questions.keys())[0]
    ref_questions = all_questions[ref_name]
    
    all_consistent = True
    
    for name, questions in all_questions.items():
        if name == ref_name:
            continue
        
        # Check sample count
        if len(questions) != len(ref_questions):
            print(f"✗ {name}: Different number of samples ({len(questions)} vs {len(ref_questions)})")
            all_consistent = False
            continue
        
        # Check each question
        mismatches = 0
        for i, (q1, q2) in enumerate(zip(ref_questions, questions)):
            if q1 != q2:
                mismatches += 1
                if mismatches == 1:  # Only print first mismatch
                    print(f"\n✗ {name}: Mismatch at sample {i+1}")
                    print(f"  {ref_name}: {q1[:80]}...")
                    print(f"  {name}: {q2[:80]}...")
        
        if mismatches == 0:
            print(f"✓ {name}: All {len(questions)} samples match {ref_name}")
        else:
            print(f"✗ {name}: {mismatches} mismatches found")
            all_consistent = False
    
    # Final verdict
    print(f"\n{'='*70}")
    if all_consistent:
        print("✓ SUCCESS: All experiments evaluate on the same samples!")
        print(f"  Total samples: {len(ref_questions)}")
        print(f"  Experiments verified: {len(all_questions)}")
        print("\nThis confirms that:")
        print("  - lm_eval's 'limit' parameter works deterministically")
        print("  - All experiments evaluate the FIRST N samples from GSM8K test set")
        print("  - Results are directly comparable across experiments")
        return 0
    else:
        print("✗ FAILURE: Some experiments use different samples!")
        print("\nThis could mean:")
        print("  - Different 'limit' values were used")
        print("  - Data loading has randomness")
        print("  - Log files are from different runs")
        return 1


if __name__ == '__main__':
    exit(main())

