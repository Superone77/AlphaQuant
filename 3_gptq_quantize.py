#!/usr/bin/env python
"""
Step 3: GPTQ Weight Quantization

This script applies GPTQ (Gradient-based Post-Training Quantization) to update
model weights according to the quantization config from step 2.

GPTQ uses Hessian information to minimize quantization error.

Usage:
    python 3_gptq_quantize.py \\
        --model allenai/OLMoE-1B-7B-0924 \\
        --config configs/auto_quant_config.json \\
        --dataset wikitext2 \\
        --save quantized_model.pt

Output:
    - Quantized model checkpoint
    - Can be evaluated in step 4
"""

import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
import logging
import torch
from pathlib import Path

from alphaquant.gptq import gptq_quantize_model, rtn_quantize_model, GPTQConfig
from alphaquant.gptq.data_utils import CalibrationDataLoader
from alphaquant.utils.replacement import load_layer_config, plan_model_layer_schemes, summarize_config


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Step 3: GPTQ weight quantization"
    )
    
    # Model settings
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model ID or local path"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Quantization config JSON (from step 2)"
    )
    
    # Dataset settings
    parser.add_argument(
        "--dataset",
        type=str,
        default="wikitext2",
        choices=["wikitext2", "c4", "ptb"],
        help="Calibration dataset"
    )
    parser.add_argument(
        "--nsamples",
        type=int,
        default=128,
        help="Number of calibration samples"
    )
    parser.add_argument(
        "--seqlen",
        type=int,
        default=2048,
        help="Sequence length for calibration"
    )
    
    # GPTQ settings
    parser.add_argument(
        "--use-gptq",
        action="store_true",
        default=True,
        help="Use GPTQ algorithm (default: True)"
    )
    parser.add_argument(
        "--use-rtn",
        action="store_true",
        help="Use RTN (Round-To-Nearest) instead of GPTQ"
    )
    parser.add_argument(
        "--groupsize",
        type=int,
        default=128,
        help="Group size for GPTQ"
    )
    parser.add_argument(
        "--actorder",
        action="store_true",
        help="Use activation order for GPTQ"
    )
    parser.add_argument(
        "--use-hadamard",
        action="store_true",
        help="Use Hadamard transform to suppress outliers (recommended for better quantization)"
    )
    
    # Output settings
    parser.add_argument(
        "--save",
        type=str,
        required=True,
        help="Path to save quantized model"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda/cpu)"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    logger.info("="*60)
    logger.info("Step 3: GPTQ Weight Quantization")
    logger.info("="*60)
    
    # Create output directory
    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load quantization config
    logger.info(f"Loading quantization config: {args.config}")
    config = load_layer_config(args.config)
    
    # Load model
    logger.info(f"Loading model: {args.model}")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    
    # Plan quantization scheme
    logger.info("Planning quantization scheme...")
    plan = plan_model_layer_schemes(model, config)
    summarize_config(plan)
    
    # Load calibration data
    logger.info(f"Loading calibration dataset: {args.dataset}")
    dataloader = CalibrationDataLoader(
        dataset_name=args.dataset,
        tokenizer=tokenizer,
        nsamples=args.nsamples,
        seqlen=args.seqlen,
        seed=42
    )
    
    # Apply quantization
    if args.use_rtn or not args.use_gptq:
        logger.info("Using RTN (Round-To-Nearest) quantization")
        quantized_model = rtn_quantize_model(model, plan)
    else:
        logger.info("Using GPTQ quantization")
        if args.use_hadamard:
            logger.info("✓ Hadamard transform enabled for outlier suppression")
        gptq_config = GPTQConfig(
            groupsize=args.groupsize,
            actorder=args.actorder,
            percdamp=0.01,
            use_hadamard=args.use_hadamard
        )
        quantized_model = gptq_quantize_model(
            model=model,
            layer_config=plan,
            dataloader=dataloader,
            gptq_config=gptq_config
        )
    
    # Save quantized model
    logger.info(f"Saving quantized model to: {args.save}")
    torch.save({
        'model_state_dict': quantized_model.state_dict(),
        'config': config,
        'plan': plan
    }, args.save)
    
    logger.info("✓ GPTQ quantization complete!")
    logger.info(f"\nNext step: Use 4_evaluate_model.py to evaluate the quantized model")


if __name__ == "__main__":
    main()

