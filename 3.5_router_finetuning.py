#!/usr/bin/env python
"""
Step 3.5: Router Finetuning

This script finetunes only the router weights of the quantized model from step 3.
All attention and expert weights are frozen, and we minimize cross-entropy loss
on the calibration dataset.

Usage:
    python 3.5_router_finetuning.py \
        --model allenai/OLMoE-1B-7B-0924 \
        --checkpoint quantized_model.pt \
        --save router_finetuned_model.pt \
        --lr 1e-4 \
        --batch_size 1 \
        --weight_decay 1e-4 \
        --num_epochs 1

Output:
    - Router-finetuned model checkpoint
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
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm

from alphaquant.gptq.data_utils import get_wikitext2
from transformers import AutoModelForCausalLM, AutoTokenizer


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Step 3.5: Router finetuning"
    )
    
    # Model settings
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model ID or local path"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Quantized model checkpoint from step 3"
    )
    
    # Dataset settings
    parser.add_argument(
        "--dataset",
        type=str,
        default="wikitext2",
        choices=["wikitext2"],
        help="Calibration dataset (default: wikitext2)"
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
    
    # Training settings
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for AdamW optimizer"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for training"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay for AdamW optimizer"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    # Output settings
    parser.add_argument(
        "--save",
        type=str,
        required=True,
        help="Path to save router-finetuned model"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda/cpu)"
    )
    
    return parser.parse_args()


def freeze_all_except_routers(model):
    """
    Freeze all model parameters except router (gate) parameters.
    
    Args:
        model: The model to freeze
        
    Returns:
        List of trainable parameter names
    """
    trainable_params = []
    frozen_params = []
    
    for name, param in model.named_parameters():
        # Only keep gate (router) parameters trainable
        if 'gate' in name.lower():
            param.requires_grad = True
            trainable_params.append(name)
        else:
            param.requires_grad = False
            frozen_params.append(name)
    
    logger.info(f"Trainable parameters ({len(trainable_params)}):")
    for name in trainable_params:
        logger.info(f"  ✓ {name}")
    
    logger.info(f"\nFrozen parameters: {len(frozen_params)}")
    
    return trainable_params


def compute_cross_entropy_loss(model, input_ids, device):
    """
    Compute cross-entropy loss for language modeling.
    
    Args:
        model: The language model
        input_ids: Input token IDs
        device: Device to use
        
    Returns:
        Cross-entropy loss
    """
    input_ids = input_ids.to(device)
    
    # Forward pass
    outputs = model(input_ids, labels=input_ids)
    
    return outputs.loss


def main():
    args = parse_args()
    
    logger.info("="*60)
    logger.info("Step 3.5: Router Finetuning")
    logger.info("="*60)
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Create output directory
    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load model
    logger.info(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    
    # Load quantized checkpoint
    logger.info(f"Loading quantized checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Freeze all parameters except routers
    logger.info("\nFreezing attention and expert weights...")
    trainable_params = freeze_all_except_routers(model)
    
    if len(trainable_params) == 0:
        logger.warning("⚠️  No trainable parameters found! Make sure the model has router/gate layers.")
        return
    
    # Setup optimizer - only optimize trainable parameters
    optimizer_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        optimizer_params,
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    logger.info(f"\nOptimizer: AdamW")
    logger.info(f"  Learning rate: {args.lr}")
    logger.info(f"  Weight decay: {args.weight_decay}")
    logger.info(f"  Trainable parameters: {sum(p.numel() for p in optimizer_params):,}")
    
    # Load calibration data
    logger.info(f"\nLoading calibration dataset: {args.dataset}")
    logger.info(f"  Samples: {args.nsamples}")
    logger.info(f"  Sequence length: {args.seqlen}")
    
    train_data = get_wikitext2(
        nsamples=args.nsamples,
        seed=args.seed,
        seqlen=args.seqlen,
        tokenizer=tokenizer
    )
    
    # Training loop
    logger.info("\n" + "="*60)
    logger.info("Starting Router Finetuning")
    logger.info("="*60)
    
    model.train()
    
    for epoch in range(args.num_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        
        total_loss = 0.0
        num_batches = 0
        
        # Create progress bar
        pbar = tqdm(range(0, len(train_data), args.batch_size), desc=f"Epoch {epoch+1}")
        
        for i in pbar:
            # Get batch
            batch_end = min(i + args.batch_size, len(train_data))
            input_ids = train_data[i:batch_end]
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Compute loss
            loss = compute_cross_entropy_loss(model, input_ids, args.device)
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            # Track loss
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Log epoch statistics
        avg_loss = total_loss / num_batches
        logger.info(f"Epoch {epoch + 1} - Average Loss: {avg_loss:.4f}")
    
    # Save finetuned model
    logger.info(f"\nSaving router-finetuned model to: {args.save}")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': checkpoint.get('config', {}),
        'plan': checkpoint.get('plan', {}),
        'router_finetuning': {
            'lr': args.lr,
            'weight_decay': args.weight_decay,
            'num_epochs': args.num_epochs,
            'batch_size': args.batch_size,
            'nsamples': args.nsamples,
            'seqlen': args.seqlen,
        }
    }, args.save)
    
    logger.info("\n" + "="*60)
    logger.info("✓ Router finetuning complete!")
    logger.info("="*60)
    logger.info(f"\nNext step: Use 4_evaluate_model.py to evaluate the finetuned model")


if __name__ == "__main__":
    main()

