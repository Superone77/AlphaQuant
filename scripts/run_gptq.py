#!/usr/bin/env python3
"""
GPTQ quantization script for AlphaQuant.

This script provides a command-line interface for quantizing models
using GPTQ with mixed-precision configurations.

Example usage:
    python scripts/run_gptq.py \\
        --model meta-llama/Llama-2-7b-hf \\
        --config configs/gptq_mixed_precision.json \\
        --dataset wikitext2 \\
        --nsamples 128 \\
        --save quantized_model.pt
"""

import sys
import os

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from alphaquant.gptq import gptq_quantize_model, rtn_quantize_model, GPTQConfig
from alphaquant.gptq.data_utils import CalibrationDataLoader
from alphaquant.utils.replacement import (
    load_layer_config,
    plan_model_layer_schemes,
    summarize_config
)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='GPTQ quantization for AlphaQuant'
    )
    
    # Model arguments
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Model name or path (HuggingFace model)'
    )
    parser.add_argument(
        '--model_type',
        type=str,
        default='auto',
        choices=['auto', 'llama', 'gpt2', 'opt', 'bloom', 'mixtral', 'qwen'],
        help='Model architecture type'
    )
    parser.add_argument(
        '--dtype',
        type=str,
        default='bfloat16',
        choices=['float32', 'float16', 'bfloat16'],
        help='Model dtype'
    )
    
    # Quantization config
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to quantization config JSON'
    )
    
    # Calibration data
    parser.add_argument(
        '--dataset',
        type=str,
        default='wikitext2',
        choices=['wikitext2', 'c4'],
        help='Calibration dataset'
    )
    parser.add_argument(
        '--nsamples',
        type=int,
        default=128,
        help='Number of calibration samples'
    )
    parser.add_argument(
        '--seqlen',
        type=int,
        default=2048,
        help='Sequence length for calibration'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Random seed'
    )
    
    # GPTQ algorithm parameters
    parser.add_argument(
        '--blocksize',
        type=int,
        default=128,
        help='GPTQ block size'
    )
    parser.add_argument(
        '--percdamp',
        type=float,
        default=0.01,
        help='GPTQ dampening percentage'
    )
    parser.add_argument(
        '--actorder',
        action='store_true',
        help='Use activation ordering in GPTQ'
    )
    parser.add_argument(
        '--static_groups',
        action='store_true',
        help='Use static groups in GPTQ'
    )
    
    # Method selection
    parser.add_argument(
        '--method',
        type=str,
        default='gptq',
        choices=['gptq', 'rtn'],
        help='Quantization method'
    )
    
    # Output
    parser.add_argument(
        '--save',
        type=str,
        default=None,
        help='Path to save quantized model'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use'
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Load model
    logger.info(f'Loading model: {args.model}')
    dtype_map = {
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16
    }
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype_map[args.dtype],
        device_map='cpu',  # Load to CPU first
        trust_remote_code=True
    )
    model.eval()
    
    # Load tokenizer
    logger.info('Loading tokenizer')
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load quantization config
    logger.info(f'Loading quantization config: {args.config}')
    layer_config_raw = load_layer_config(args.config)
    logger.info(summarize_config(layer_config_raw))
    
    # Plan layer-wise schemes
    logger.info('Planning layer-wise quantization schemes')
    plans = plan_model_layer_schemes(
        model,
        layer_config_raw,
        target_module_classes=('Linear',)
    )
    logger.info(f'Total layers to quantize: {len(plans)}')
    
    # Convert to dict for easier lookup
    layer_config = {name: scheme for name, scheme in plans}
    
    # Quantize
    if args.method == 'gptq':
        # Load calibration data
        logger.info(f'Loading calibration data: {args.dataset}')
        dataloader = CalibrationDataLoader(
            dataset_name=args.dataset,
            nsamples=args.nsamples,
            seed=args.seed,
            seqlen=args.seqlen,
            tokenizer=tokenizer
        )
        
        # Create GPTQ config
        gptq_config = GPTQConfig(
            blocksize=args.blocksize,
            percdamp=args.percdamp,
            actorder=args.actorder,
            static_groups=args.static_groups
        )
        
        # Run GPTQ
        quantizers = gptq_quantize_model(
            model=model,
            dataloader=dataloader,
            layer_config=layer_config,
            device=args.device,
            gptq_config=gptq_config,
            model_type=args.model_type,
            dtype=args.dtype
        )
    else:  # rtn
        quantizers = rtn_quantize_model(
            model=model,
            layer_config=layer_config,
            device=args.device,
            model_type=args.model_type,
            dtype=args.dtype
        )
    
    # Save if requested
    if args.save:
        logger.info(f'Saving quantized model to: {args.save}')
        save_dict = {
            'model': model.state_dict(),
            'quantizers': quantizers,
            'config': layer_config_raw,
            'args': vars(args)
        }
        torch.save(save_dict, args.save)
        logger.info('Model saved successfully')
    
    logger.info('Quantization complete!')


if __name__ == '__main__':
    main()

