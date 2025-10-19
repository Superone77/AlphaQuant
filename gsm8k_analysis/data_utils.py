"""
GSM8K data utilities for calibration and evaluation.

This module provides functions to load GSM8K dataset for calibration and evaluation.
"""

import sys
import os
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import random
from typing import Optional, List
import torch
from datasets import load_dataset
from transformers import PreTrainedTokenizer


def get_gsm8k_calibration(
    nsamples: int = 128,
    seed: int = 0,
    seqlen: int = 2048,
    tokenizer: Optional[PreTrainedTokenizer] = None,
    split: str = 'train'
) -> torch.Tensor:
    """
    Get GSM8K calibration data.
    
    Args:
        nsamples: Number of samples to use
        seed: Random seed
        seqlen: Sequence length
        tokenizer: Tokenizer to use
        split: Dataset split ('train' or 'test')
        
    Returns:
        Tokenized data tensor
    """
    if tokenizer is None:
        raise ValueError("Tokenizer must be provided")
    
    # Load GSM8K dataset
    dataset = load_dataset('gsm8k', 'main', split=split)
    
    random.seed(seed)
    samples = []
    
    # Sample from dataset
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    
    for idx in indices[:nsamples]:
        item = dataset[idx]
        # Combine question and answer for calibration
        text = f"Question: {item['question']}\nAnswer: {item['answer']}"
        
        # Tokenize
        encoded = tokenizer(text, return_tensors='pt', max_length=seqlen, truncation=True)
        inp = encoded.input_ids
        
        # Pad if needed
        if inp.shape[1] < seqlen:
            pad_len = seqlen - inp.shape[1]
            inp = torch.cat([inp, torch.zeros((1, pad_len), dtype=inp.dtype)], dim=1)
        
        samples.append(inp[:, :seqlen])
    
    return torch.cat(samples, dim=0)


def get_gsm8k_test_samples(
    nsamples: int = 256,
    seed: int = 0,
    split: str = 'test'
) -> List[int]:
    """
    Get indices of GSM8K test samples for evaluation.
    
    Args:
        nsamples: Number of samples to select
        seed: Random seed
        split: Dataset split
        
    Returns:
        List of sample indices
    """
    dataset = load_dataset('gsm8k', 'main', split=split)
    
    random.seed(seed)
    indices = list(range(min(nsamples, len(dataset))))
    
    return indices


class GSM8KCalibrationDataLoader:
    """
    Data loader wrapper for GSM8K calibration.
    """
    
    def __init__(
        self,
        nsamples: int = 128,
        seed: int = 0,
        seqlen: int = 2048,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        split: str = 'train'
    ):
        """
        Initialize GSM8K calibration data loader.
        
        Args:
            nsamples: Number of samples
            seed: Random seed
            seqlen: Sequence length
            tokenizer: Tokenizer
            split: Dataset split
        """
        self.nsamples = nsamples
        self.seed = seed
        self.seqlen = seqlen
        self.tokenizer = tokenizer
        self.split = split
        self._data = None
    
    def _load_data(self):
        """Load data lazily."""
        if self._data is None:
            self._data = get_gsm8k_calibration(
                self.nsamples, self.seed, self.seqlen, self.tokenizer, self.split
            )
    
    def __iter__(self):
        """Iterate over samples."""
        self._load_data()
        for i in range(self._data.shape[0]):
            yield self._data[i:i+1]
    
    def __len__(self):
        """Get number of samples."""
        return self.nsamples

