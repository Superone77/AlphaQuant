"""
Evaluation utilities for saving detailed sample-level results.

This module provides functions to save detailed evaluation results including
questions, model answers, and reasoning processes.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif hasattr(obj, 'dtype'):
            # Handle numpy dtypes
            return str(obj)
        return super(NumpyEncoder, self).default(obj)


def save_detailed_samples(
    results: Dict[str, Any],
    output_path: str,
    task_name: str = "gsm8k"
):
    """
    Save detailed sample-level results from lm_eval.
    
    Args:
        results: Results dictionary from lm_eval
        output_path: Path to save the detailed samples JSON
        task_name: Name of the task (default: gsm8k)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Extract samples if available
    samples = []
    if 'samples' in results and task_name in results['samples']:
        raw_samples = results['samples'][task_name]
        
        for idx, sample in enumerate(raw_samples):
            sample_data = {
                'index': idx,
                'question': sample.get('doc', {}).get('question', ''),
                'gold_answer': sample.get('doc', {}).get('answer', ''),
                'model_output': sample.get('resps', [['']])[0][0] if 'resps' in sample else '',
                'model_answer': sample.get('filtered_resps', [''])[0] if 'filtered_resps' in sample else '',
                'correct': sample.get('exact_match,strict-match', False),
                'doc_id': sample.get('doc_id', idx),
                'arguments': sample.get('arguments', [])
            }
            samples.append(sample_data)
    
    # Save to JSON
    detailed_results = {
        'timestamp': datetime.now().isoformat(),
        'task': task_name,
        'total_samples': len(samples),
        'samples': samples
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    
    print(f"Saved {len(samples)} detailed samples to: {output_path}")
    
    return len(samples)


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


def save_samples_with_reasoning(
    results: Dict[str, Any],
    output_path: str,
    task_name: str = "gsm8k"
):
    """
    Save detailed samples with extracted reasoning process.
    
    Args:
        results: Results dictionary from lm_eval
        output_path: Path to save the detailed samples JSON
        task_name: Name of the task
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    samples = []
    if 'samples' in results and task_name in results['samples']:
        raw_samples = results['samples'][task_name]
        
        for idx, sample in enumerate(raw_samples):
            # Get model output
            model_output = ''
            if 'resps' in sample and len(sample['resps']) > 0:
                if len(sample['resps'][0]) > 0:
                    model_output = sample['resps'][0][0]
            
            # Extract reasoning
            reasoning_data = extract_reasoning_from_output(model_output)
            
            sample_data = {
                'index': idx,
                'doc_id': sample.get('doc_id', idx),
                'question': sample.get('doc', {}).get('question', ''),
                'gold_answer': sample.get('doc', {}).get('answer', ''),
                'model_reasoning': reasoning_data['reasoning'],
                'model_final_answer': reasoning_data['final_answer'],
                'model_full_output': reasoning_data['full_output'],
                'correct': sample.get('exact_match,strict-match', None),
                'metrics': {
                    key: value for key, value in sample.items()
                    if key not in ['doc', 'resps', 'filtered_resps', 'arguments']
                }
            }
            samples.append(sample_data)
    
    # Calculate statistics
    correct_count = sum(1 for s in samples if s.get('correct', False))
    accuracy = correct_count / len(samples) if samples else 0
    
    detailed_results = {
        'timestamp': datetime.now().isoformat(),
        'task': task_name,
        'statistics': {
            'total_samples': len(samples),
            'correct': correct_count,
            'incorrect': len(samples) - correct_count,
            'accuracy': accuracy
        },
        'samples': samples
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    
    print(f"\n📊 Detailed sample results:")
    print(f"  - Total samples: {len(samples)}")
    print(f"  - Correct: {correct_count}")
    print(f"  - Accuracy: {accuracy:.4f}")
    print(f"  - Saved to: {output_path}")
    
    return detailed_results

