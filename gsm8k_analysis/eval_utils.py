"""
Evaluation utilities for logging detailed sample-level results.

This module provides functions to log detailed evaluation results including
questions, model answers, and reasoning processes to text files.
"""

from pathlib import Path
from typing import Dict, Any
from datetime import datetime


def log_samples_to_file(
    results: Dict[str, Any],
    output_path: str,
    task_name: str = "gsm8k"
):
    """
    Log detailed sample-level results to a text file.
    
    Args:
        results: Results dictionary from lm_eval
        output_path: Path to save the log file
        task_name: Name of the task (default: gsm8k)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Extract samples if available
    if 'samples' not in results or task_name not in results['samples']:
        print(f"Warning: No samples found for task {task_name}")
        return 0
    
    raw_samples = results['samples'][task_name]
    
    # Write to log file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write(f"GSM8K Evaluation Samples - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Samples: {len(raw_samples)}\n")
        f.write("=" * 100 + "\n\n")
        
        correct_count = 0
        
        for idx, sample in enumerate(raw_samples):
            # Extract information
            question = sample.get('doc', {}).get('question', '')
            gold_answer = sample.get('doc', {}).get('answer', '')
            
            # Get model output
            model_output = ''
            if 'resps' in sample and len(sample['resps']) > 0:
                if len(sample['resps'][0]) > 0:
                    model_output = str(sample['resps'][0][0])
            
            # Check correctness
            is_correct = sample.get('exact_match,strict-match', False)
            if is_correct:
                correct_count += 1
            
            # Extract reasoning and answer
            reasoning_data = extract_reasoning_from_output(model_output)
            
            # Write to log
            f.write(f"{'='*100}\n")
            f.write(f"Sample #{idx + 1}\n")
            f.write(f"{'='*100}\n\n")
            
            f.write(f"Question:\n{question}\n\n")
            
            f.write(f"Gold Answer:\n{gold_answer}\n\n")
            
            f.write(f"Model Reasoning:\n{reasoning_data['reasoning']}\n\n")
            
            f.write(f"Model Final Answer:\n{reasoning_data['final_answer']}\n\n")
            
            f.write(f"Correct: {'✓ YES' if is_correct else '✗ NO'}\n\n")
            
            f.write(f"Full Model Output:\n{'-'*50}\n{model_output}\n{'-'*50}\n\n")
        
        # Summary at the end
        f.write("=" * 100 + "\n")
        f.write("SUMMARY\n")
        f.write("=" * 100 + "\n")
        f.write(f"Total Samples: {len(raw_samples)}\n")
        f.write(f"Correct: {correct_count}\n")
        f.write(f"Incorrect: {len(raw_samples) - correct_count}\n")
        f.write(f"Accuracy: {correct_count / len(raw_samples):.4f} ({correct_count}/{len(raw_samples)})\n")
    
    print(f"\n📝 Logged {len(raw_samples)} samples to: {output_path}")
    print(f"   Correct: {correct_count}/{len(raw_samples)} ({correct_count/len(raw_samples):.2%})")
    
    return len(raw_samples)


def extract_reasoning_from_output(output: str) -> Dict[str, str]:
    """
    Extract reasoning process from model output.
    
    Args:
        output: Raw model output
        
    Returns:
        Dictionary with reasoning and final answer
    """
    # Try to split reasoning and answer
    # Common patterns in GSM8K:
    # - "#### number" marks the final answer
    # - Everything before is reasoning
    
    if '####' in output:
        parts = output.split('####')
        reasoning = parts[0].strip()
        answer = parts[1].strip() if len(parts) > 1 else ''
    else:
        reasoning = output.strip()
        answer = ''
    
    return {
        'reasoning': reasoning,
        'final_answer': answer,
        'full_output': output
    }


def log_samples_with_reasoning(
    results: Dict[str, Any],
    output_path: str,
    task_name: str = "gsm8k"
):
    """
    Log detailed samples with extracted reasoning process to a text file.
    
    Args:
        results: Results dictionary from lm_eval
        output_path: Path to save the log file
        task_name: Name of the task
    """
    return log_samples_to_file(results, output_path, task_name)

