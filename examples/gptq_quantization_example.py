#!/usr/bin/env python3
"""
Example script demonstrating GPTQ quantization with AlphaQuant.

This script shows how to:
1. Load a model and configure mixed-precision quantization
2. Run GPTQ quantization
3. Save and load the quantized model
4. Evaluate the model
"""

import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from alphaquant.gptq import gptq_quantize_model, GPTQConfig
from alphaquant.gptq.data_utils import CalibrationDataLoader
from alphaquant.utils.replacement import (
    load_layer_config,
    plan_model_layer_schemes,
    summarize_config
)


def main():
    """Main example function."""
    
    # ===== Configuration =====
    MODEL_NAME = "meta-llama/Llama-2-7b-hf"  # Change to your model
    CONFIG_PATH = "configs/gptq_mixed_precision.json"
    DATASET = "wikitext2"
    NSAMPLES = 128
    SEQLEN = 2048
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 80)
    print("AlphaQuant GPTQ Quantization Example")
    print("=" * 80)
    
    # ===== Step 1: Load Model =====
    print("\n[Step 1] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map='cpu',  # Start on CPU
        trust_remote_code=True
    )
    model.eval()
    print(f"Model loaded: {MODEL_NAME}")
    
    # ===== Step 2: Load Tokenizer =====
    print("\n[Step 2] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer loaded")
    
    # ===== Step 3: Load Quantization Config =====
    print("\n[Step 3] Loading quantization configuration...")
    layer_config_raw = load_layer_config(CONFIG_PATH)
    print(summarize_config(layer_config_raw))
    
    # Plan layer-wise schemes
    plans = plan_model_layer_schemes(
        model,
        layer_config_raw,
        target_module_classes=('Linear',)
    )
    layer_config = {name: scheme for name, scheme in plans}
    print(f"Total layers to quantize: {len(plans)}")
    
    # ===== Step 4: Prepare Calibration Data =====
    print("\n[Step 4] Preparing calibration data...")
    dataloader = CalibrationDataLoader(
        dataset_name=DATASET,
        nsamples=NSAMPLES,
        seed=0,
        seqlen=SEQLEN,
        tokenizer=tokenizer
    )
    print(f"Calibration data: {DATASET}, {NSAMPLES} samples")
    
    # ===== Step 5: Configure GPTQ =====
    print("\n[Step 5] Configuring GPTQ...")
    gptq_config = GPTQConfig(
        blocksize=128,
        percdamp=0.01,
        actorder=False,  # Set to True for better accuracy (slower)
        static_groups=False
    )
    print(f"GPTQ Config: {gptq_config}")
    
    # ===== Step 6: Run GPTQ Quantization =====
    print("\n[Step 6] Running GPTQ quantization...")
    print("This may take a while depending on model size and number of samples...")
    
    quantizers = gptq_quantize_model(
        model=model,
        dataloader=dataloader,
        layer_config=layer_config,
        device=DEVICE,
        gptq_config=gptq_config,
        model_type='auto',
        dtype='bfloat16'
    )
    
    print(f"\nQuantization complete! Quantized {len(quantizers)} layers")
    
    # ===== Step 7: Save Quantized Model =====
    print("\n[Step 7] Saving quantized model...")
    save_path = "quantized_model.pt"
    save_dict = {
        'model': model.state_dict(),
        'quantizers': quantizers,
        'config': layer_config_raw,
        'model_name': MODEL_NAME
    }
    torch.save(save_dict, save_path)
    print(f"Model saved to: {save_path}")
    
    # ===== Step 8: Test Generation (Optional) =====
    print("\n[Step 8] Testing generation...")
    test_prompt = "The future of AI is"
    inputs = tokenizer(test_prompt, return_tensors="pt").to(DEVICE)
    model = model.to(DEVICE)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=50,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Input: {test_prompt}")
    print(f"Output: {generated_text}")
    
    # ===== Summary =====
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Model: {MODEL_NAME}")
    print(f"Config: {CONFIG_PATH}")
    print(f"Quantized layers: {len(quantizers)}")
    print(f"Saved to: {save_path}")
    print("\nTo load the quantized model:")
    print("  checkpoint = torch.load('quantized_model.pt')")
    print("  model.load_state_dict(checkpoint['model'])")
    print("=" * 80)


if __name__ == '__main__':
    main()

