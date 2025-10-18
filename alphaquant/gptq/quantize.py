"""
Main GPTQ quantization pipeline.

Provides high-level functions for quantizing models with mixed-precision GPTQ.
"""

import logging
from typing import Dict, Any, Optional, Iterator
import torch
import torch.nn as nn
from tqdm import tqdm

from .gptq import GPTQ, GPTQConfig
from .model_utils import find_layers, get_layers_for_model, cleanup_memory
from ..utils.replacement import create_quantizer_from_scheme


logger = logging.getLogger(__name__)


@torch.no_grad()
def gptq_quantize_model(
    model: nn.Module,
    dataloader: Iterator[torch.Tensor],
    layer_config: Dict[str, Dict[str, Any]],
    device: str = 'cuda',
    gptq_config: Optional[GPTQConfig] = None,
    model_type: str = 'auto',
    dtype: str = 'bfloat16'
) -> Dict[str, Any]:
    """
    Quantize a model using GPTQ with mixed-precision configuration.
    
    This is the main entry point for GPTQ quantization in AlphaQuant.
    It supports layer-wise mixed-precision quantization based on a configuration.
    
    Args:
        model: The model to quantize (e.g., HuggingFace model)
        dataloader: Calibration data iterator
        layer_config: Layer-wise quantization configuration
                     Maps layer names to quantization schemes
        device: Device to use for quantization
        gptq_config: GPTQ algorithm configuration
        model_type: Model architecture type ('llama', 'gpt2', 'auto', etc.)
        dtype: Data type for computation
        
    Returns:
        Dictionary with quantization metadata (scales, zeros, etc.)
    """
    if gptq_config is None:
        gptq_config = GPTQConfig()
    
    logger.info('===== Starting GPTQ Quantization =====')
    
    # Disable caching during quantization
    use_cache = model.config.use_cache if hasattr(model.config, 'use_cache') else False
    if hasattr(model.config, 'use_cache'):
        model.config.use_cache = False
    
    # Get transformer layers
    layers = get_layers_for_model(model, model_type)
    
    # Move embedding and normalization layers to device
    if hasattr(model, 'model'):
        if hasattr(model.model, 'embed_tokens'):
            model.model.embed_tokens = model.model.embed_tokens.to(device)
        if hasattr(model.model, 'norm'):
            model.model.norm = model.model.norm.to(device)
    
    # Move first layer to device
    layers[0] = layers[0].to(device)
    
    # Capture inputs to first layer
    dtype_model = next(iter(model.parameters())).dtype
    cache = {'i': 0, 'attention_mask': None, 'position_ids': None}
    
    # Estimate input tensor size
    # We'll collect a few samples to determine the shape
    sample_inputs = []
    for batch in dataloader:
        if isinstance(batch, dict):
            sample_inputs.append(batch)
        else:
            sample_inputs.append(batch)
        if len(sample_inputs) >= 2:
            break
    
    # Determine hidden size and sequence length from model config
    hidden_size = model.config.hidden_size
    # Try to get sequence length from first batch
    first_batch = sample_inputs[0]
    if isinstance(first_batch, dict) and 'input_ids' in first_batch:
        seqlen = first_batch['input_ids'].shape[1]
    else:
        seqlen = first_batch.shape[1] if len(first_batch.shape) > 1 else 2048
    
    # Count actual number of samples
    nsamples = 0
    for _ in dataloader:
        nsamples += 1
    
    inps = torch.zeros((nsamples, seqlen, hidden_size), dtype=dtype_model, device=device)
    
    cache = {'i': 0, 'attention_mask': None, 'position_ids': None, 'position_embeddings': None}
    
    # Catcher to intercept inputs
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs.get('attention_mask', None)
            cache['position_ids'] = kwargs.get('position_ids', None)
            cache['position_embeddings'] = kwargs.get('position_embeddings', None)
            raise ValueError  # Stop forward pass
    
    layers[0] = Catcher(layers[0])
    
    # Collect inputs
    for batch in dataloader:
        try:
            if isinstance(batch, dict):
                batch = {k: v.to(device) for k, v in batch.items()}
                model(**batch)
            else:
                model(batch.to(device))
        except ValueError:
            pass
    
    layers[0] = layers[0].module
    torch.cuda.empty_cache()
    
    # Prepare outputs
    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    position_embeddings = cache['position_embeddings']
    
    quantizers = {}
    
    # Quantize each layer
    for i in tqdm(range(len(layers)), desc='GPTQ Layers'):
        logger.info(f'\nLayer {i}:')
        layer = layers[i].to(device)
        
        # Find all linear layers in this transformer layer
        full = find_layers(layer, layers=[nn.Linear])
        
        # Group layers for sequential processing
        # Process each linear layer
        for name in full.keys():
            full_name = f'model.layers.{i}.{name}'
            logger.info(f'  {name}')
            
            # Get quantization scheme for this layer
            if full_name not in layer_config:
                logger.warning(f'    No config for {full_name}, skipping')
                continue
            
            scheme = layer_config[full_name]
            
            # Skip if marked to skip
            if scheme.get('skip', False):
                logger.info(f'    Skipping {full_name}')
                continue
            
            # Create quantizer for this layer
            (WQ, WCfg), (AQ, ACfg) = create_quantizer_from_scheme(scheme, dtype)
            
            # Extract parameters
            group_size = scheme.get('group_size', -1)
            extra = scheme.get('extra', {})
            
            # Build quantizer config
            w_kwargs = {'group_size': group_size, 'dtype': dtype}
            if 'format' in extra:
                w_kwargs['format'] = extra['format']
            
            weight_quantizer = WQ(WCfg(**w_kwargs))
            
            # Create GPTQ instance
            use_hadamard = gptq_config.use_hadamard if gptq_config else False
            gptq = GPTQ(full[name], use_hadamard=use_hadamard)
            
            # Register hook to collect inputs
            def add_batch(name_):
                def hook(module, inp, out):
                    gptq.add_batch(inp[0].data, out.data if out is not None else None)
                return hook
            
            handle = full[name].register_forward_hook(add_batch(name))
            
            # Forward pass to collect Hessian
            for j in range(nsamples):
                # Prepare forward kwargs
                forward_kwargs = {
                    'attention_mask': attention_mask,
                }
                if position_ids is not None:
                    forward_kwargs['position_ids'] = position_ids
                if position_embeddings is not None:
                    forward_kwargs['position_embeddings'] = position_embeddings
                
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    **forward_kwargs
                )[0]
            
            handle.remove()
            
            # Apply GPTQ quantization
            gptq.fasterquant(
                quantizer=weight_quantizer,
                blocksize=gptq_config.blocksize,
                percdamp=gptq_config.percdamp,
                groupsize=group_size,
                actorder=gptq_config.actorder,
                static_groups=gptq_config.static_groups
            )
            
            # Store quantizer
            quantizers[full_name] = weight_quantizer
            gptq.free()
        
        # Forward pass with quantized weights to get outputs for next layer
        for j in range(nsamples):
            # Prepare forward kwargs
            forward_kwargs = {
                'attention_mask': attention_mask,
            }
            if position_ids is not None:
                forward_kwargs['position_ids'] = position_ids
            if position_embeddings is not None:
                forward_kwargs['position_embeddings'] = position_embeddings
            
            outs[j] = layer(
                inps[j].unsqueeze(0),
                **forward_kwargs
            )[0]
        
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        
        # Swap inputs and outputs for next layer
        inps, outs = outs, inps
    
    # Restore cache setting
    if hasattr(model.config, 'use_cache'):
        model.config.use_cache = use_cache
    
    cleanup_memory(verbose=True)
    logger.info('===== GPTQ Quantization Complete =====\n')
    
    return quantizers


@torch.no_grad()
def rtn_quantize_model(
    model: nn.Module,
    layer_config: Dict[str, Dict[str, Any]],
    device: str = 'cuda',
    model_type: str = 'auto',
    dtype: str = 'bfloat16'
) -> Dict[str, Any]:
    """
    Quantize a model using RTN (Round-to-Nearest).
    
    RTN is a simpler quantization method that doesn't require calibration data.
    
    Args:
        model: The model to quantize
        layer_config: Layer-wise quantization configuration
        device: Device to use
        model_type: Model architecture type
        dtype: Data type for computation
        
    Returns:
        Dictionary with quantization metadata
    """
    logger.info('===== Starting RTN Quantization =====')
    
    layers = get_layers_for_model(model, model_type)
    quantizers = {}
    
    for i in tqdm(range(len(layers)), desc='RTN Layers'):
        layer = layers[i].to(device)
        
        # Find all linear layers
        subset = find_layers(layer, layers=[nn.Linear])
        
        for name in subset:
            full_name = f'model.layers.{i}.{name}'
            
            if full_name not in layer_config:
                continue
            
            scheme = layer_config[full_name]
            
            if scheme.get('skip', False):
                continue
            
            # Create quantizer
            (WQ, WCfg), _ = create_quantizer_from_scheme(scheme, dtype)
            
            group_size = scheme.get('group_size', -1)
            extra = scheme.get('extra', {})
            
            w_kwargs = {'group_size': group_size, 'dtype': dtype}
            if 'format' in extra:
                w_kwargs['format'] = extra['format']
            
            weight_quantizer = WQ(WCfg(**w_kwargs))
            
            # Quantize weights directly
            W = subset[name].weight.data
            quantized = weight_quantizer.quantize_weight(W)
            
            if isinstance(quantized, tuple):
                subset[name].weight.data = quantized[0].to(W.dtype)
            else:
                subset[name].weight.data = quantized.to(W.dtype)
            
            quantizers[full_name] = weight_quantizer
        
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
    
    cleanup_memory(verbose=True)
    logger.info('===== RTN Quantization Complete =====\n')
    
    return quantizers

