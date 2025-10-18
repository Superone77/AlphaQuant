# AlphaQuant

## 🆕 GPTQ Mixed-Precision Quantization

AlphaQuant now includes a complete GPTQ pipeline for mixed-precision post-training quantization!

### Quick Start with GPTQ

```bash
# Using the CLI script
python scripts/run_gptq.py \
  --model meta-llama/Llama-2-7b-hf \
  --config configs/gptq_mixed_precision.json \
  --dataset wikitext2 \
  --nsamples 128 \
  --save quantized_model.pt

# Or use the example script
python examples/gptq_quantization_example.py
```

### Features
- ✅ Mixed-precision: INT2/3/4/6/8, FP4/6/8, MXFP4/6/8
- ✅ GPTQ algorithm with Hessian-based optimization
- ✅ JSON-based configuration for easy layer-wise quantization
- ✅ Support for MoE models (OLMoE, Mixtral, Qwen-MoE)
- ✅ RTN fallback for quick quantization

**📖 Full documentation**: See [GPTQ_PIPELINE_README.md](GPTQ_PIPELINE_README.md)

---

## Basic Quantization & Calibration

### 1) Quantize & calibrate

```bash
python scripts/quantize_model.py \
  --model meta-llama/Llama-3.2-1B \
  --wq mxfp4 --aq mxfp8 --group 64 \
  --include q_proj k_proj v_proj o_proj up_proj gate_proj down_proj \
  --calib_ds wikitext --calib_split wikitext-2-raw-v1 --calib_batches 16 \
  --save ./llama32_1b_mx48_g64
```

### 2) Evaluate with lm-eval
```bash
python scripts/eval_with_lm_eval.py \
  --pretrained ./llama32_1b_mx48_g64 \
  --tasks hellaswag,arc_easy,winogrande \
  --batch_size 8
```

## Layer-wise JSON config
You can control which layers use which quantization scheme via a JSON file passed with `--layer-config`.

Example JSON (`configs/llama32_1b_mx48.json`):

```json
{
  "default": {"wq": "mxfp8", "aq": "mxfp8", "group_size": 128},
  "overrides": [
    {"pattern": "model.layers.0.*", "wq": "mxfp4"},
    {"pattern": "model.layers.*.q_proj", "wq": "mxfp4", "group_size": 64}
  ]
}
```

- **pattern**: glob pattern matched against names from `model.named_modules()`, e.g., `model.layers.12.q_proj`.
- **default**: base scheme applied to all matched target modules before overrides.
- Any extra fields (e.g., `block_size`) are passed through for your scheme implementation to consume.

Dry-run to preview the plan (no model load required):

```bash
python scripts/quantize_model.py \
  --layer-config configs/llama32_1b_mx48.json \
  --dry-run --save-plan ./plans/llama32_1b_mx48_preview.json
```

Expand the plan against a specific model (loads the model):

```bash
python scripts/quantize_model.py \
  --model meta-llama/Llama-3.2-1B \
  --device cpu \
  --dtype fp32 \
  --layer-config configs/llama32_1b_mx48.json \
  --save-plan ./plans/llama32_1b_mx48_expanded.json
```

Note: This only produces a mapping plan. Hook your actual replacement/quantization logic where needed (e.g., apply the plan to replace `nn.Linear` with `QuantLinear`).

### Notes
- You can also set `"skip": true` in an override to explicitly exclude matched modules from quantization in the planning stage.
- An OLMoE-oriented preset is available at `configs/olmoe_mixed_quant.json`:
  - Skip all `*.mlp.gate*` and `*lm_head*`
  - Quantize all attention layers with MXFP8
  - Inside `*.experts.*`, set `{gate_proj,up_proj,down_proj}` to MXFP4

Example usage:
```bash
python scripts/quantize_model.py \
  --model allenai/OLMoE-1B \
  --device cuda \
  --dtype bf16 \
  --layer-config configs/olmoe_mixed_quant.json \
  --save-plan ./plans/olmoe_mixed_expanded.json
```
```
export CUDA_VISIBLE_DEVICES=3

python scripts/alpha_hill_quantization.py \
    --model "/local/mnt2/workspace/wanqi/tmp/LLM-Research/OLMoE-1B-7B-0924-Instruct" \
    --mxfp4-ratio 0.3 \
    --output-config "results/quant_alpha_quant.json" \
    --output-csv "results/alpha_hill_results.csv"

python scripts/quantize_model.py \
  --model /local/mnt2/workspace/wanqi/tmp/LLM-Research/OLMoE-1B-7B-0924-Instruct \
  --device cuda \
  --dtype bf16 \
  --layer-config results/quant_alpha_quant.json \
  --save-plan ./plans/olmoe_mixed_alpha.json

python scripts/eval_with_plans.py \
  --model /local/mnt2/workspace/wanqi/tmp/LLM-Research/OLMoE-1B-7B-0924-Instruct \
  --tasks gsm8k \
  --plan ./plans/olmoe_mxfp4.json \
  --batch_size 32
```