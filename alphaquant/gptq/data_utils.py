"""
Data utilities for GPTQ calibration.

Provides data loaders for calibration datasets.
"""

import random
from typing import Iterator, Optional
import torch
from datasets import load_dataset
from transformers import PreTrainedTokenizer


def get_wikitext2(
    nsamples: int = 128,
    seed: int = 0,
    seqlen: int = 2048,
    tokenizer: Optional[PreTrainedTokenizer] = None
) -> torch.Tensor:
    """
    Get WikiText-2 calibration data.
    
    Args:
        nsamples: Number of samples to use
        seed: Random seed
        seqlen: Sequence length
        tokenizer: Tokenizer to use
        
    Returns:
        Tokenized data tensor
    """
    if tokenizer is None:
        raise ValueError("Tokenizer must be provided")
    
    # Load dataset
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    # Tokenize
    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    # Sample calibration data
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        trainloader.append(inp)
    
    return torch.cat(trainloader, dim=0)


def get_c4(
    nsamples: int = 128,
    seed: int = 0,
    seqlen: int = 2048,
    tokenizer: Optional[PreTrainedTokenizer] = None
) -> torch.Tensor:
    """
    Get C4 calibration data.
    
    Args:
        nsamples: Number of samples to use
        seed: Random seed
        seqlen: Sequence length
        tokenizer: Tokenizer to use
        
    Returns:
        Tokenized data tensor
    """
    if tokenizer is None:
        raise ValueError("Tokenizer must be provided")
    
    # Load C4 dataset
    traindata = load_dataset(
        'allenai/c4',
        data_files={'train': 'en/c4-train.00000-of-01024.json.gz'},
        split='train'
    )
    
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        trainloader.append(inp)
    
    return torch.cat(trainloader, dim=0)


def get_loaders(
    name: str,
    nsamples: int = 128,
    seed: int = 0,
    seqlen: int = 2048,
    tokenizer: Optional[PreTrainedTokenizer] = None
) -> Iterator[torch.Tensor]:
    """
    Get calibration data loader.
    
    Args:
        name: Dataset name ('wikitext2', 'c4', etc.)
        nsamples: Number of samples
        seed: Random seed
        seqlen: Sequence length
        tokenizer: Tokenizer
        
    Returns:
        Iterator over data batches
    """
    if name == 'wikitext2':
        data = get_wikitext2(nsamples, seed, seqlen, tokenizer)
    elif name == 'c4':
        data = get_c4(nsamples, seed, seqlen, tokenizer)
    else:
        raise ValueError(f"Unknown dataset: {name}")
    
    # Yield one sample at a time
    for i in range(data.shape[0]):
        yield data[i:i+1]


class CalibrationDataLoader:
    """
    Data loader wrapper for calibration.
    
    Provides batched access to calibration data.
    """
    
    def __init__(
        self,
        dataset_name: str = 'wikitext2',
        nsamples: int = 128,
        seed: int = 0,
        seqlen: int = 2048,
        tokenizer: Optional[PreTrainedTokenizer] = None
    ):
        """
        Initialize calibration data loader.
        
        Args:
            dataset_name: Name of dataset
            nsamples: Number of samples
            seed: Random seed
            seqlen: Sequence length
            tokenizer: Tokenizer
        """
        self.dataset_name = dataset_name
        self.nsamples = nsamples
        self.seed = seed
        self.seqlen = seqlen
        self.tokenizer = tokenizer
        self._data = None
    
    def _load_data(self):
        """Load data lazily."""
        if self._data is None:
            if self.dataset_name == 'wikitext2':
                self._data = get_wikitext2(
                    self.nsamples, self.seed, self.seqlen, self.tokenizer
                )
            elif self.dataset_name == 'c4':
                self._data = get_c4(
                    self.nsamples, self.seed, self.seqlen, self.tokenizer
                )
            else:
                raise ValueError(f"Unknown dataset: {self.dataset_name}")
    
    def __iter__(self):
        """Iterate over samples."""
        self._load_data()
        for i in range(self._data.shape[0]):
            yield self._data[i:i+1]
    
    def __len__(self):
        """Get number of samples."""
        return self.nsamples

