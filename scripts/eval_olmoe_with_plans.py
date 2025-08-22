#!/usr/bin/env python
from __future__ import annotations
import argparse
import json
import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
import torch
from alphaquant.utils.replacement import load_layer_config, plan_model_layer_schemes
from alphaquant.modules.quant_linear import QuantLinearConfig
from alphaquant.quantizers.mxfp4 import MXFP4Quantizer, MXFP4Config
from alphaquant.quantizers.mxfp8 import MXFP8Quantizer, MXFP8Config


def load_quantization_plan(plan_path: str) -> Dict[str, Dict[str, Any]]:
    """Load quantization plan from JSON file."""
    with open(plan_path, 'r') as f:
        plan = json.load(f)
    return plan


def create_quantizer_from_scheme(scheme: Dict[str, Any], dtype: str) -> Tuple[Any, Any]:
    """Create quantizer class and config from scheme dict."""
    wq_scheme = scheme.get('wq', 'mxfp8')
    aq_scheme = scheme.get('aq', 'mxfp8')
    group_size = scheme.get('group_size', 32)
    
    # Map scheme names to quantizer classes
    quantizer_map = {
        'mxfp4': (MXFP4Quantizer, MXFP4Config),
        'mxfp8': (MXFP8Quantizer, MXFP8Config),
    }
    
    if wq_scheme not in quantizer_map:
        raise ValueError(f"Unknown weight quantization scheme: {wq_scheme}")
    if aq_scheme not in quantizer_map:
        raise ValueError(f"Unknown activation quantization scheme: {aq_scheme}")
    
    WQ, WCfg = quantizer_map[wq_scheme]
    AQ, ACfg = quantizer_map[aq_scheme]
    
    return (WQ, WCfg), (AQ, ACfg)


def apply_layer_wise_quantization(model: Any, plan: Dict[str, Dict[str, Any]], dtype: str) -> List[str]:
    """Apply layer-wise quantization based on the plan."""
    replaced_modules = []
    
    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and module_name in plan:
            scheme = plan[module_name]
            
            # Create quantizer configs
            (WQ, WCfg), (AQ, ACfg) = create_quantizer_from_scheme(scheme, dtype)
            
            # Create quantization config
            qcfg = QuantLinearConfig(
                weight_quantizer_cls=WQ,
                weight_quantizer_cfg=WCfg(group_size=scheme.get('group_size', 32), dtype=dtype),
                act_quantizer_cls=AQ,
                act_quantizer_cfg=ACfg(group_size=scheme.get('group_size', 32), dtype=dtype),
                bias=True
            )
            
            # Replace the module
            parent_name = module_name.rsplit('.', 1)[0] if '.' in module_name else ''
            attr = module_name.split('.')[-1]
            parent = model.get_submodule(parent_name) if parent_name else model
            
            # Create QuantLinear replacement
            from alphaquant.modules.quant_linear import QuantLinear
            qlin = QuantLinear(module.in_features, module.out_features, bias=module.bias is not None)
            if module.bias is not None:
                qlin.inner.bias.data = module.bias.data.clone()
            qlin.inner.weight.data = module.weight.data.clone()
            
            setattr(parent, attr, qlin)
            replaced_modules.append(module_name)
    
    return replaced_modules


def main():
    parser = argparse.ArgumentParser(description="Evaluate OLMoE with layer-wise quantization from plans")
    parser.add_argument('--model', required=True, help='Path to OLMoE model or HF model ID')
    parser.add_argument('--plan', required=True, help='Path to quantization plan JSON file')
    parser.add_argument('--tasks', default='gsm8k', help='Comma-separated list of tasks (default: gsm8k)')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for evaluation')
    parser.add_argument('--device', default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--dtype', default='bfloat16', help='Model dtype (bfloat16/float16/float32)')
    parser.add_argument('--dry_run', action='store_true', help='Only show quantization plan without applying')
    
    args = parser.parse_args()
    
    # Load quantization plan
    print(f"Loading quantization plan from: {args.plan}")
    plan = load_quantization_plan(args.plan)
    print(f"Loaded plan with {len(plan)} modules")
    
    if args.dry_run:
        print("Dry run mode - showing first 10 modules in plan:")
        for i, (name, scheme) in enumerate(list(plan.items())[:10]):
            print(f"  {name}: {scheme}")
        if len(plan) > 10:
            print(f"  ... and {len(plan) - 10} more modules")
        return
    
    # Build HFLM
    print(f"Loading model: {args.model}")
    lm = HFLM(pretrained=args.model, device=args.device, dtype=args.dtype, batch_size=args.batch_size)
    model = lm.model
    
    # Apply quantization
    print("Applying layer-wise quantization...")
    replaced = apply_layer_wise_quantization(model, plan, args.dtype)
    print(f"Quantized {len(replaced)} modules")
    
    # Run evaluation
    print(f"Running evaluation on tasks: {args.tasks}")
    results = evaluator.simple_evaluate(
        model=lm,
        tasks=args.tasks.split(','),
        batch_size=args.batch_size,
    )
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(results)
    
    # Save results
    output_file = f"olmoe_quantized_{args.tasks.replace(',', '_')}_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main() 