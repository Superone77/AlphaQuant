"""
Model utilities for GPTQ quantization.

Provides functions for finding layers, managing model structure, etc.
"""

from typing import List, Dict, Any, Type, Tuple
import torch
import torch.nn as nn


def find_layers(
    module: nn.Module,
    layers: List[Type[nn.Module]] = None,
    name: str = ''
) -> Dict[str, nn.Module]:
    """
    Recursively find all layers of specified types in a module.
    
    Args:
        module: Module to search
        layers: List of layer types to find (default: [nn.Linear])
        name: Current module name (for recursion)
        
    Returns:
        Dictionary mapping layer names to layer modules
    """
    if layers is None:
        layers = [nn.Linear]
    
    if type(module) in layers:
        return {name: module}
    
    res = {}
    for name1, child in module.named_children():
        res.update(
            find_layers(
                child,
                layers=layers,
                name=name + '.' + name1 if name != '' else name1
            )
        )
    return res


def get_layers_for_model(model: nn.Module, model_type: str = 'auto') -> List[nn.Module]:
    """
    Get the transformer layers of a model.
    
    Args:
        model: The model
        model_type: Model type hint ('llama', 'gpt2', 'opt', 'auto')
        
    Returns:
        List of transformer layers
    """
    # Auto-detect model type
    if model_type == 'auto':
        model_name = type(model).__name__.lower()
        if 'llama' in model_name:
            model_type = 'llama'
        elif 'gpt' in model_name:
            model_type = 'gpt2'
        elif 'opt' in model_name:
            model_type = 'opt'
        elif 'bloom' in model_name:
            model_type = 'bloom'
        elif 'mixtral' in model_name:
            model_type = 'mixtral'
        elif 'qwen' in model_name:
            model_type = 'qwen'
        else:
            # Try to find layers attribute
            if hasattr(model, 'model') and hasattr(model.model, 'layers'):
                return model.model.layers
            elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
                return model.transformer.h
            else:
                raise ValueError(
                    f"Could not auto-detect model type for {type(model).__name__}. "
                    "Please specify model_type manually."
                )
    
    # Get layers based on type
    if model_type in ['llama', 'mixtral', 'qwen', 'opt']:
        return model.model.layers
    elif model_type == 'gpt2':
        return model.transformer.h
    elif model_type == 'bloom':
        return model.transformer.h
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_sequential_groups(model_type: str = 'llama') -> List[List[str]]:
    """
    Get sequential groups of layers for a model type.
    
    Sequential groups define which layers can be quantized together.
    
    Args:
        model_type: Type of model
        
    Returns:
        List of layer name groups
    """
    if model_type in ['llama', 'mistral']:
        return [
            ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj'],
            ['self_attn.o_proj'],
            ['mlp.gate_proj', 'mlp.up_proj'],
            ['mlp.down_proj']
        ]
    elif model_type == 'mixtral':
        return [
            ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj'],
            ['self_attn.o_proj'],
            # MoE layers handled specially
        ]
    elif model_type == 'qwen':
        return [
            ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj'],
            ['self_attn.o_proj'],
            # MoE layers handled specially
        ]
    elif model_type == 'gpt2':
        return [
            ['attn.c_attn'],
            ['attn.c_proj'],
            ['mlp.c_fc'],
            ['mlp.c_proj']
        ]
    else:
        # Default: return empty, will quantize all layers individually
        return []


def cleanup_memory(verbose: bool = False):
    """
    Clean up GPU memory.
    
    Args:
        verbose: Whether to print memory stats
    """
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    if verbose and torch.cuda.is_available():
        print(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated")
        print(f"GPU memory: {torch.cuda.memory_reserved() / 1e9:.2f} GB reserved")


def get_module_by_name(model: nn.Module, module_name: str) -> Tuple[nn.Module, str, nn.Module]:
    """
    Get a module by its full name path.
    
    Args:
        model: The model
        module_name: Full module name (e.g., 'model.layers.0.self_attn.q_proj')
        
    Returns:
        Tuple of (parent_module, attribute_name, module)
    """
    atoms = module_name.split('.')
    parent = model
    
    for atom in atoms[:-1]:
        parent = getattr(parent, atom)
    
    return parent, atoms[-1], getattr(parent, atoms[-1])


def set_module_by_name(model: nn.Module, module_name: str, new_module: nn.Module):
    """
    Replace a module by its full name path.
    
    Args:
        model: The model
        module_name: Full module name
        new_module: The new module to set
    """
    parent, attr_name, _ = get_module_by_name(model, module_name)
    setattr(parent, attr_name, new_module)

