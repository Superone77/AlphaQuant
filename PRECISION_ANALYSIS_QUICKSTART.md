# 🚀 Precision Analysis Quick Start

## 📋 Overview

Analyze how different numerical precisions (FP32, FP16, BF16) affect Alpha-Hill values across all layers in your model. This helps identify precision-sensitive layers for optimal mixed-precision quantization strategies.

---

## ⚡ Quick Start

### Option 1: Basic Command (CPU)

```bash
python scripts/analyze_precision_alpha_hill.py \
  --model meta-llama/Llama-3.1-8B \
  --precisions fp32,fp16,bf16 \
  --output-dir ./results/llama31_precision
```

### Option 2: With CUDA (Recommended)

```bash
export CUDA_VISIBLE_DEVICES=0

python scripts/analyze_precision_alpha_hill.py \
  --model meta-llama/Llama-3.1-8B \
  --device cuda \
  --load-dtype bf16 \
  --precisions fp32,fp16,bf16 \
  --output-dir ./results/llama31_precision
```

### Option 3: Using Example Script

```bash
# Test with small model (Llama-3.2-1B)
python scripts/example_precision_analysis_simple.py

# Or use bash script for comprehensive analysis
bash scripts/example_llama31_precision_analysis.sh
```

---

## 📊 What You Get

### 1. CSV Files

**`precision_alpha_hill_results.csv`** - Detailed per-layer results:
```
layer_name,category,out_features,in_features,alpha_fp32,alpha_fp16,alpha_bf16,...
model.layers.0.self_attn.q_proj,attn_q,4096,4096,2.456,2.453,2.455,...
model.layers.0.self_attn.k_proj,attn_k,4096,4096,2.381,2.378,2.380,...
...
```

**`precision_summary_statistics.csv`** - Statistical summary:
```
precision,count,mean,std,min,median,max,q25,q75
fp32,290,2.567,0.234,2.123,2.543,3.456,2.401,2.698
fp16,290,2.563,0.235,2.119,2.540,3.450,2.398,2.695
bf16,290,2.565,0.234,2.121,2.542,3.453,2.400,2.697
```

### 2. Visualizations

```
results/llama31_precision/
└── plots/
    ├── alpha_distribution_by_precision.png      # Box plot overview
    ├── category_attn_q.png                      # Grouped by layer type
    ├── category_mlp_up.png
    ├── combined_batch_1.png                     # Combined bar charts
    └── individual_layers/                       # One plot per layer
        ├── layer_001_*.png
        ├── layer_002_*.png
        └── ...
```

---

## 🎯 Common Use Cases

### 1. Analyze Specific Layer Types

**Attention layers only:**
```bash
python scripts/analyze_precision_alpha_hill.py \
  --model meta-llama/Llama-3.1-8B \
  --filter-layers ".*(q_proj|k_proj|v_proj|o_proj).*" \
  --precisions fp32,fp16,bf16 \
  --output-dir ./results/attention_precision
```

**MLP layers only:**
```bash
python scripts/analyze_precision_alpha_hill.py \
  --model meta-llama/Llama-3.1-8B \
  --filter-layers ".*(up_proj|gate_proj|down_proj).*" \
  --precisions fp32,fp16,bf16 \
  --output-dir ./results/mlp_precision
```

**First 5 layers only (for testing):**
```bash
python scripts/analyze_precision_alpha_hill.py \
  --model meta-llama/Llama-3.1-8B \
  --filter-layers ".*layers\\.[0-4]\\..*" \
  --precisions fp32,fp16,bf16 \
  --output-dir ./results/first5_precision
```

### 2. Extended Precision Analysis

```bash
# Include FP64 for reference
python scripts/analyze_precision_alpha_hill.py \
  --model meta-llama/Llama-3.1-8B \
  --precisions fp64,fp32,fp16,bf16 \
  --output-dir ./results/extended_precision
```

### 3. High-Resolution Outputs

```bash
# Generate PDF plots for publications
python scripts/analyze_precision_alpha_hill.py \
  --model meta-llama/Llama-3.1-8B \
  --precisions fp32,fp16,bf16 \
  --plot-format pdf \
  --output-dir ./results/hires_precision
```

---

## 🔍 Interpreting Results

### Alpha-Hill Values

- **α > 3.0**: High concentration in top singular values
  - **Interpretation**: Layer is **precision-sensitive**
  - **Action**: Use higher precision (FP16/BF16)

- **2.5 < α < 3.0**: Moderate concentration
  - **Interpretation**: Layer has **medium sensitivity**
  - **Action**: BF16 or mixed precision

- **α < 2.5**: More uniform distribution
  - **Interpretation**: Layer is **robust to quantization**
  - **Action**: Can use lower precision (INT8/FP8)

### Precision Sensitivity

Compare α values across precisions:

```python
# Example layer results
Layer: model.layers.0.self_attn.q_proj
  fp32: α = 2.567
  fp16: α = 2.565
  bf16: α = 2.566
  
Variation: 0.002 (0.08%)  →  LOW sensitivity ✓
```

```python
# High sensitivity example
Layer: model.layers.15.mlp.gate_proj
  fp32: α = 3.145
  fp16: α = 3.089
  bf16: α = 3.142
  
Variation: 0.056 (1.78%)  →  HIGH sensitivity ⚠️
```

### Using Results for Quantization

1. **High variation across precisions** → Keep in higher precision
2. **Low α AND low variation** → Aggressive quantization (INT4/FP4)
3. **High α OR high variation** → Conservative quantization (FP8/BF16)

---

## 💡 Pro Tips

### Performance Optimization

```bash
# Fastest: Use CUDA + BF16 loading
--device cuda --load-dtype bf16

# Reduce computation: Lower k-frac
--k-frac 0.05  # Use 5% of eigenvalues instead of 10%

# Analyze subset: Filter specific layers
--filter-layers ".*layers\\.(0|15|31)\\..*"  # First, middle, last
```

### Memory Management

```bash
# For large models on limited memory:
--device cpu --load-dtype fp32 --max-layers-per-plot 5
```

### Batch Processing

```bash
# Analyze multiple models
for MODEL in "meta-llama/Llama-3.1-8B" "meta-llama/Llama-3.2-1B"; do
  python scripts/analyze_precision_alpha_hill.py \
    --model "$MODEL" \
    --precisions fp32,fp16,bf16 \
    --output-dir "./results/$(basename $MODEL)_precision"
done
```

---

## 📈 Example Workflow

### Step 1: Quick Test
```bash
# Test with first 3 layers
python scripts/analyze_precision_alpha_hill.py \
  --model meta-llama/Llama-3.1-8B \
  --filter-layers ".*layers\\.[0-2]\\..*" \
  --precisions fp32,fp16,bf16 \
  --output-dir ./results/test
```

### Step 2: Analyze by Type
```bash
# Attention
python scripts/analyze_precision_alpha_hill.py \
  --model meta-llama/Llama-3.1-8B \
  --filter-layers ".*(q_proj|k_proj|v_proj|o_proj).*" \
  --output-dir ./results/attention

# MLP  
python scripts/analyze_precision_alpha_hill.py \
  --model meta-llama/Llama-3.1-8B \
  --filter-layers ".*(up_proj|gate_proj|down_proj).*" \
  --output-dir ./results/mlp
```

### Step 3: Full Analysis
```bash
# Complete model
python scripts/analyze_precision_alpha_hill.py \
  --model meta-llama/Llama-3.1-8B \
  --precisions fp32,fp16,bf16 \
  --device cuda \
  --load-dtype bf16 \
  --output-dir ./results/full
```

### Step 4: Generate Quantization Config
```python
import pandas as pd

# Load results
df = pd.read_csv('results/full/precision_alpha_hill_results.csv')

# Identify precision-sensitive layers
df['alpha_var'] = df[['alpha_fp32', 'alpha_fp16', 'alpha_bf16']].std(axis=1)
df['alpha_mean'] = df[['alpha_fp32', 'alpha_fp16', 'alpha_bf16']].mean(axis=1)

# High sensitivity or high alpha → keep higher precision
sensitive_layers = df[(df['alpha_var'] > 0.01) | (df['alpha_mean'] > 3.0)]

print("Layers requiring high precision:")
print(sensitive_layers[['layer_name', 'alpha_mean', 'alpha_var']])
```

---

## 🐛 Troubleshooting

### Issue: Out of Memory
**Solution:**
```bash
--device cpu --load-dtype fp32 --filter-layers ".*layers\\.[0-5]\\..*"
```

### Issue: NaN values in results
**Solution:**
```bash
--k-frac 0.15  # Increase for better numerical stability
```

### Issue: Slow execution
**Solution:**
```bash
--device cuda --load-dtype bf16 --k-frac 0.05
```

---

## 📚 Related Documentation

- **Detailed Guide**: `scripts/README_precision_analysis.md`
- **Testing**: `scripts/test_precision_analysis.py`
- **Examples**: `scripts/example_precision_analysis_simple.py`

---

## 🎓 Citation

If you use this tool in your research, please cite:

```bibtex
@software{alphaquant2024,
  title={AlphaQuant: Precision-Aware Mixed-Precision Quantization},
  year={2024},
  url={https://github.com/yourusername/AlphaQuant}
}
```

---

## ✅ Quick Checklist

Before running analysis:
- [ ] Model is accessible (local path or HF hub)
- [ ] GPU available? Set `--device cuda`
- [ ] Enough disk space for outputs (~500MB for Llama 8B)
- [ ] Dependencies installed (`torch`, `transformers`, `matplotlib`, `pandas`)

After running:
- [ ] Check CSV for numerical results
- [ ] Review distribution plot for overview
- [ ] Identify outliers (high α or high variation)
- [ ] Generate quantization strategy based on findings

---

**Need help?** Check `scripts/README_precision_analysis.md` for comprehensive documentation.

