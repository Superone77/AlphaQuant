#!/usr/bin/env python
"""
Example: Router Finetuning Workflow

This example demonstrates the complete workflow including router finetuning:
1. Compute Alpha-Hill values
2. Allocate bitwidth based on sensitivity
3. Apply GPTQ quantization
3.5. Finetune router weights (NEW!)
4. Evaluate the model

This is particularly useful for heavily quantized MoE models where
routing decisions need to adapt to quantized expert weights.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from alphaquant.alpha_hill.utils import alpha_hill_from_model
from alphaquant.gptq import gptq_quantize_model, GPTQConfig
from alphaquant.gptq.data_utils import CalibrationDataLoader, get_wikitext2
from alphaquant.utils.replacement import load_layer_config, plan_model_layer_schemes


def compute_alpha_values(model, output_path="results/alpha_values.csv"):
    """Step 1: Compute Alpha-Hill sensitivity values."""
    print("\n" + "="*60)
    print("Step 1: Computing Alpha-Hill Values")
    print("="*60)
    
    alpha_values = alpha_hill_from_model(model)
    
    # Save to CSV
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write("layer_name,alpha_value\n")
        for name, value in alpha_values.items():
            f.write(f"{name},{value}\n")
    
    print(f"✓ Alpha values saved to {output_path}")
    return alpha_values


def allocate_bitwidth(alpha_values, mxfp4_ratio=0.3, output_path="configs/auto_quant_config.json"):
    """Step 2: Allocate bitwidth based on sensitivity."""
    print("\n" + "="*60)
    print("Step 2: Allocating Bitwidth")
    print("="*60)
    
    # Sort layers by alpha (ascending)
    sorted_layers = sorted(alpha_values.items(), key=lambda x: x[1])
    
    # Calculate split point
    num_mxfp4 = int(len(sorted_layers) * mxfp4_ratio)
    
    # Create config
    config = {
        "default": {
            "wq": "mxfp8",
            "aq": "bf16",
            "group_size": 128
        },
        "overrides": []
    }
    
    # Assign high precision to sensitive layers (low alpha)
    for i, (layer_name, alpha) in enumerate(sorted_layers):
        if i < num_mxfp4:
            config["overrides"].append({
                "pattern": layer_name,
                "wq": "mxfp4",
                "comment": f"Sensitive layer (alpha={alpha:.4f})"
            })
    
    # Save config
    import json
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✓ Config saved to {output_path}")
    print(f"  MXFP4 layers: {num_mxfp4}")
    print(f"  MXFP8 layers: {len(sorted_layers) - num_mxfp4}")
    
    return config


def gptq_quantize(model, config, tokenizer, output_path="results/quantized_model.pt"):
    """Step 3: Apply GPTQ quantization."""
    print("\n" + "="*60)
    print("Step 3: GPTQ Quantization")
    print("="*60)
    
    # Plan quantization scheme
    plan = plan_model_layer_schemes(model, config)
    
    # Load calibration data
    dataloader = CalibrationDataLoader(
        dataset_name="wikitext2",
        tokenizer=tokenizer,
        nsamples=128,
        seqlen=2048,
        seed=42
    )
    
    # Apply GPTQ
    gptq_config = GPTQConfig(
        groupsize=128,
        actorder=False,
        percdamp=0.01
    )
    
    quantized_model = gptq_quantize_model(
        model=model,
        layer_config=plan,
        dataloader=dataloader,
        config=gptq_config
    )
    
    # Save checkpoint
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict': quantized_model.state_dict(),
        'config': config,
        'plan': plan
    }, output_path)
    
    print(f"✓ Quantized model saved to {output_path}")
    return quantized_model


def router_finetuning(model, tokenizer, checkpoint_path="results/quantized_model.pt",
                     output_path="outputs/router_finetuned_model.pt",
                     lr=1e-4, num_epochs=1):
    """Step 3.5: Finetune router weights."""
    print("\n" + "="*60)
    print("Step 3.5: Router Finetuning")
    print("="*60)
    
    # Load quantized checkpoint
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Freeze all except routers
    trainable_params = []
    for name, param in model.named_parameters():
        if 'gate' in name.lower():
            param.requires_grad = True
            trainable_params.append(name)
        else:
            param.requires_grad = False
    
    print(f"Trainable parameters: {len(trainable_params)}")
    for name in trainable_params:
        print(f"  ✓ {name}")
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=1e-4
    )
    
    # Load training data
    train_data = get_wikitext2(
        nsamples=128,
        seed=42,
        seqlen=2048,
        tokenizer=tokenizer
    )
    
    # Training loop
    model.train()
    device = next(model.parameters()).device
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        total_loss = 0.0
        num_batches = 0
        
        for i in range(len(train_data)):
            input_ids = train_data[i:i+1].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if (i + 1) % 20 == 0:
                print(f"  Batch {i+1}/{len(train_data)} - Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1} - Average Loss: {avg_loss:.4f}")
    
    # Save finetuned model
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': checkpoint.get('config', {}),
        'plan': checkpoint.get('plan', {}),
        'router_finetuning': {
            'lr': lr,
            'num_epochs': num_epochs,
        }
    }, output_path)
    
    print(f"✓ Router-finetuned model saved to {output_path}")
    return model


def main():
    """Run the complete workflow."""
    print("="*60)
    print("Router Finetuning Workflow Example")
    print("="*60)
    
    # Configuration
    model_name = "allenai/OLMoE-1B-7B-0924"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mxfp4_ratio = 0.3  # 30% high precision
    
    print(f"\nModel: {model_name}")
    print(f"Device: {device}")
    print(f"MXFP4 Ratio: {mxfp4_ratio}")
    
    # Load model
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Step 1: Compute Alpha values
    alpha_values = compute_alpha_values(model)
    
    # Step 2: Allocate bitwidth
    config = allocate_bitwidth(alpha_values, mxfp4_ratio=mxfp4_ratio)
    
    # Step 3: GPTQ quantization
    quantized_model = gptq_quantize(model, config, tokenizer)
    
    # Step 3.5: Router finetuning (NEW!)
    finetuned_model = router_finetuning(
        model=quantized_model,
        tokenizer=tokenizer,
        lr=1e-4,
        num_epochs=1
    )
    
    print("\n" + "="*60)
    print("✓ Complete workflow finished!")
    print("="*60)
    print("\nNext steps:")
    print("1. Evaluate the finetuned model:")
    print("   python 4_evaluate_model.py \\")
    print("       --model allenai/OLMoE-1B-7B-0924 \\")
    print("       --checkpoint outputs/router_finetuned_model.pt \\")
    print("       --tasks hellaswag,arc_easy,winogrande")
    print("\n2. Compare with quantized-only model:")
    print("   python 4_evaluate_model.py \\")
    print("       --model allenai/OLMoE-1B-7B-0924 \\")
    print("       --checkpoint results/quantized_model.pt \\")
    print("       --tasks hellaswag,arc_easy,winogrande")


if __name__ == "__main__":
    main()

