# Precision-Based Alpha-Hill Analysis

## Overview

This script (`analyze_precision_alpha_hill.py`) computes and visualizes Alpha-Hill values for each layer of a model across different numerical precisions. This helps understand how quantization precision affects the eigenvalue distribution characteristics of weight matrices.

## Purpose

- **Research Tool**: Analyze the impact of different precisions (FP32, FP16, BF16, etc.) on Alpha-Hill metrics
- **Quantization Planning**: Identify which layers are more sensitive to precision changes
- **Model Compression**: Guide mixed-precision quantization strategies based on precision sensitivity

## Features

- ✅ Compute Alpha-Hill values at multiple precisions per layer
- ✅ Generate individual bar charts for each layer
- ✅ Create grouped comparison plots by category
- ✅ Export results to CSV with detailed statistics
- ✅ Summary statistics across all layers and precisions
- ✅ Distribution analysis with box plots

## Usage

### Basic Usage

```bash
python scripts/analyze_precision_alpha_hill.py \
  --model meta-llama/Llama-3.1-8B \
  --precisions fp32,fp16,bf16 \
  --output-dir ./results/llama3.1_8b_precision_analysis
```

### With CUDA

```bash
python scripts/analyze_precision_alpha_hill.py \
  --model meta-llama/Llama-3.1-8B \
  --device cuda \
  --load-dtype bf16 \
  --precisions fp32,fp16,bf16 \
  --output-dir ./results/llama3.1_8b_precision_analysis
```

### Filter Specific Layers

```bash
# Only analyze attention layers
python scripts/analyze_precision_alpha_hill.py \
  --model meta-llama/Llama-3.1-8B \
  --filter-layers ".*attn.*" \
  --precisions fp32,fp16,bf16 \
  --output-dir ./results/llama3.1_8b_attn_only
```

### High-Resolution Plots

```bash
python scripts/analyze_precision_alpha_hill.py \
  --model meta-llama/Llama-3.1-8B \
  --precisions fp32,fp16,bf16 \
  --plot-format pdf \
  --max-layers-per-plot 20 \
  --output-dir ./results/llama3.1_8b_hires
```

## Arguments

### Required

- `--model`: HuggingFace model ID or local path (e.g., `meta-llama/Llama-3.1-8B`)

### Optional

- `--device`: Device for model loading (default: `cpu`)
  - Options: `cpu`, `cuda`, `cuda:0`, etc.

- `--load-dtype`: Initial dtype for loading model (default: `fp32`)
  - Options: `fp32`, `fp16`, `bf16`

- `--precisions`: Comma-separated list of precisions to analyze (default: `fp32,fp16,bf16`)
  - Options: `fp32`, `fp16`, `bf16`, `fp64`

- `--output-dir`: Output directory for results (default: `./results/precision_analysis`)

- `--k-frac`: Fraction of eigenvalues for Alpha-Hill (default: `0.1`)

- `--filter-layers`: Regex pattern to filter layers (optional)
  - Example: `".*q_proj.*"` for query projections only

- `--plot-format`: Output format for plots (default: `png`)
  - Options: `png`, `pdf`, `svg`

- `--max-layers-per-plot`: Max layers per combined plot (default: `10`, 0 for all)

- `--log-level`: Logging verbosity (default: `INFO`)
  - Options: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`

## Output Structure

```
results/llama3.1_8b_precision_analysis/
├── precision_alpha_hill_results.csv          # Main results CSV
├── precision_summary_statistics.csv          # Summary stats by precision
└── plots/
    ├── alpha_distribution_by_precision.png   # Box plot distribution
    ├── category_attn_q.png                   # Grouped by category
    ├── category_attn_k.png
    ├── category_mlp_up.png
    ├── combined_batch_1.png                  # Batched combined plots
    ├── combined_batch_2.png
    └── individual_layers/                    # One plot per layer
        ├── layer_001_model_layers_0_self_attn_q_proj.png
        ├── layer_002_model_layers_0_self_attn_k_proj.png
        └── ...
```

## CSV Format

### Main Results (`precision_alpha_hill_results.csv`)

| Column | Description |
|--------|-------------|
| `layer_name` | Full layer name from model |
| `category` | Layer category (attn_q, mlp_up, etc.) |
| `out_features` | Output dimension |
| `in_features` | Input dimension |
| `numel` | Total number of parameters |
| `alpha_fp32` | Alpha-Hill value at FP32 precision |
| `k_used_fp32` | Number of eigenvalues used (FP32) |
| `n_eigs_fp32` | Total eigenvalues computed (FP32) |
| `method_fp32` | SVD method used (FP32) |
| `alpha_fp16` | Alpha-Hill value at FP16 precision |
| ... | (repeated for each precision) |

### Summary Statistics (`precision_summary_statistics.csv`)

| Column | Description |
|--------|-------------|
| `precision` | Precision name (fp32, fp16, bf16) |
| `count` | Number of valid layers |
| `mean` | Mean Alpha-Hill value |
| `std` | Standard deviation |
| `min` | Minimum value |
| `median` | Median value |
| `max` | Maximum value |
| `q25` | 25th percentile |
| `q75` | 75th percentile |

## Interpretation

### Alpha-Hill Values

- **Higher α**: Layer has more concentrated energy in top singular values
  - More sensitive to quantization
  - May require higher precision

- **Lower α**: More uniform eigenvalue distribution
  - Less sensitive to quantization
  - Can tolerate lower precision

### Precision Sensitivity

Compare α values across precisions:
- **Large variation**: Layer is precision-sensitive
- **Small variation**: Layer is robust to precision changes

### Use Cases

1. **Mixed-Precision Planning**: Use high precision for layers with large α variations
2. **Quantization Boundaries**: Identify "safe" vs "risky" layers for aggressive quantization
3. **Hardware Mapping**: Match precision-insensitive layers to low-precision hardware units

## Examples

### Example 1: Full Llama 3.1 8B Analysis

```bash
export CUDA_VISIBLE_DEVICES=0

python scripts/analyze_precision_alpha_hill.py \
  --model meta-llama/Llama-3.1-8B \
  --device cuda \
  --load-dtype bf16 \
  --precisions fp32,fp16,bf16 \
  --k-frac 0.1 \
  --output-dir ./results/llama31_8b_full \
  --max-layers-per-plot 15 \
  --plot-format png
```

### Example 2: Attention-Only Analysis

```bash
python scripts/analyze_precision_alpha_hill.py \
  --model meta-llama/Llama-3.1-8B \
  --filter-layers ".*(q_proj|k_proj|v_proj|o_proj).*" \
  --precisions fp32,fp16,bf16 \
  --output-dir ./results/llama31_8b_attention
```

### Example 3: MLP-Only with Extended Precisions

```bash
python scripts/analyze_precision_alpha_hill.py \
  --model meta-llama/Llama-3.1-8B \
  --filter-layers ".*(up_proj|gate_proj|down_proj).*" \
  --precisions fp64,fp32,fp16,bf16 \
  --output-dir ./results/llama31_8b_mlp
```

## Performance Tips

1. **Use CUDA**: Much faster for large models
   ```bash
   --device cuda --load-dtype bf16
   ```

2. **Filter Layers**: Analyze specific layer types to save time
   ```bash
   --filter-layers ".*layers\.(0|1|2)\..*"  # First 3 layers only
   ```

3. **Reduce k-frac**: Use fewer eigenvalues for faster computation
   ```bash
   --k-frac 0.05  # Use 5% instead of 10%
   ```

4. **Batch Plots**: Control memory usage with smaller plot batches
   ```bash
   --max-layers-per-plot 5
   ```

## Troubleshooting

### Out of Memory

- Use `--device cpu` for model loading
- Reduce `--k-frac` value
- Filter specific layers with `--filter-layers`

### NaN Values

- Some precisions may produce numerical instabilities
- Try increasing `--k-frac` for better numerical stability
- Check if layer has degenerate dimensions

### Slow Performance

- Use CUDA: `--device cuda`
- Load model in lower precision: `--load-dtype bf16`
- Reduce number of precisions to analyze

## Related Scripts

- `alpha_hill_quantization.py`: Use Alpha-Hill for quantization planning
- `quantize_model.py`: Apply layer-wise quantization
- `eval_with_plans.py`: Evaluate quantized models

## Citation

If you use this analysis tool in your research, please cite the AlphaQuant framework.

