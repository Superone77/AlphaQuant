# GSM8K Analysis Pipeline

This directory contains scripts for analyzing OLMoE model's accuracy loss on the GSM8K dataset under different quantization methods.

## Overview

The pipeline evaluates the OLMoE model on 256 GSM8K samples using:
1. **Baseline**: Original (non-quantized) model
2. **MXFP4**: All expert layers quantized to MXFP4
3. **GPTQ + WikiText2**: GPTQ quantization with WikiText2 calibration data
4. **GPTQ + GSM8K**: GPTQ quantization with GSM8K training set calibration data

## Project Structure

```
gsm8k_analysis/
├── README.md                  # This file
├── run_pipeline.py            # Main pipeline orchestrator
├── run_pipeline.sh            # Bash wrapper for easy execution
├── data_utils.py              # GSM8K data loading utilities
├── eval_baseline.py           # Baseline evaluation script
├── quantize_mxfp4.py          # MXFP4 quantization script
├── quantize_gptq.py           # GPTQ quantization script
├── analyze_results.py         # Results analysis and comparison
├── configs/                   # Generated quantization plans
│   ├── mxfp4_plan_*.json
│   ├── gptq_wikitext2_plan_*.json
│   └── gptq_gsm8k_plan_*.json
└── results/                   # Evaluation results
    ├── baseline_*.json
    ├── mxfp4_*.json
    ├── gptq_wikitext2_*.json
    ├── gptq_gsm8k_*.json
    ├── pipeline_summary_*.json
    ├── comparison_*.csv
    └── comparison_*.txt
```

## Quick Start

### Run Complete Pipeline

```bash
# Using bash wrapper (recommended)
cd /Users/superone77/Code/AlphaQuant
./gsm8k_analysis/run_pipeline.sh

# Or using Python directly
python gsm8k_analysis/run_pipeline.py \
    --model allenai/OLMoE-1B-7B-0924 \
    --num_samples 256 \
    --num_calibration_samples 128 \
    --device cuda
```

### Run Individual Steps

```bash
# Step 1: Baseline evaluation
python gsm8k_analysis/eval_baseline.py \
    --model allenai/OLMoE-1B-7B-0924 \
    --num_samples 256 \
    --device cuda

# Step 2: MXFP4 quantization
python gsm8k_analysis/quantize_mxfp4.py \
    --model allenai/OLMoE-1B-7B-0924 \
    --num_samples 256 \
    --device cuda

# Step 3: GPTQ with WikiText2
python gsm8k_analysis/quantize_gptq.py \
    --model allenai/OLMoE-1B-7B-0924 \
    --calibration_data wikitext2 \
    --num_calibration_samples 128 \
    --num_eval_samples 256 \
    --device cuda

# Step 4: GPTQ with GSM8K
python gsm8k_analysis/quantize_gptq.py \
    --model allenai/OLMoE-1B-7B-0924 \
    --calibration_data gsm8k \
    --num_calibration_samples 128 \
    --num_eval_samples 256 \
    --device cuda

# Step 5: Analyze results
python gsm8k_analysis/analyze_results.py \
    --summary gsm8k_analysis/results/pipeline_summary_<timestamp>.json
```

## Pipeline Details

### 1. Baseline Evaluation (`eval_baseline.py`)

Evaluates the original OLMoE model without any quantization.

**Key Parameters:**
- `--model`: Model name or path
- `--num_samples`: Number of GSM8K test samples (default: 256)
- `--batch_size`: Evaluation batch size (default: 1)
- `--device`: Device to use (cuda/cpu)
- `--dtype`: Model dtype (bfloat16/float16/float32)

**Output:**
- JSON file with evaluation results
- Baseline accuracy for comparison

### 2. MXFP4 Quantization (`quantize_mxfp4.py`)

Quantizes all expert layers to MXFP4 format and evaluates on GSM8K.

**Features:**
- Automatic detection of expert layers
- MXFP4 quantization with e8m0 format
- Group size: 128
- Saves quantization plan for reproducibility

**Output:**
- Quantized model evaluation results
- Quantization plan JSON

### 3. GPTQ Quantization (`quantize_gptq.py`)

Applies GPTQ quantization with configurable calibration data.

**Key Parameters:**
- `--calibration_data`: Dataset for calibration (wikitext2/gsm8k/c4)
- `--num_calibration_samples`: Number of calibration samples (default: 128)
- `--bits`: Quantization bits (default: 4)
- `--percdamp`: GPTQ damping parameter (default: 0.01)
- `--blocksize`: GPTQ block size (default: 128)
- `--actorder`: Enable activation ordering

**Output:**
- Quantized model evaluation results
- Quantization plan JSON
- Optional: Quantized model checkpoint

### 4. Results Analysis (`analyze_results.py`)

Compares results from all experiments and generates analysis reports.

**Features:**
- Accuracy comparison table
- Accuracy drop calculation
- Best method identification
- Calibration data comparison

**Output:**
- Comparison CSV file
- Detailed analysis TXT file
- Key findings summary

## Example Output

```
================================================
Comparison Table
================================================
Experiment         Accuracy  Stderr  Calibration  Quantized Layers  Bits
baseline           0.6523    0.0142  N/A          N/A               N/A
gptq_gsm8k         0.6341    0.0145  gsm8k        192               4
gptq_wikitext2     0.6187    0.0146  wikitext2    192               4
mxfp4              0.5892    0.0148  N/A          192               4

================================================
Accuracy Drop from Baseline
================================================
gptq_gsm8k          : -0.0182 (-2.79%)
gptq_wikitext2      : -0.0336 (-5.15%)
mxfp4               : -0.0631 (-9.67%)

Key Findings:
- GPTQ with GSM8K calibration performs best among quantization methods
- Using task-specific calibration (GSM8K) reduces accuracy loss by 45% vs generic (WikiText2)
- MXFP4 shows higher accuracy loss due to lack of calibration
```

## Configuration Options

### Pipeline Arguments

```python
--model                    # Model name or path (default: allenai/OLMoE-1B-7B-0924)
--num_samples              # GSM8K test samples (default: 256)
--num_calibration_samples  # Calibration samples (default: 128)
--batch_size               # Evaluation batch size (default: 1)
--device                   # Device (default: cuda)
--dtype                    # Model dtype (default: bfloat16)
--output_dir               # Results directory (default: gsm8k_analysis/results)

# Skip options
--skip_baseline            # Skip baseline evaluation
--skip_mxfp4               # Skip MXFP4 quantization
--skip_gptq_wikitext       # Skip GPTQ with WikiText2
--skip_gptq_gsm8k          # Skip GPTQ with GSM8K
```

## Dependencies

All scripts use the existing AlphaQuant infrastructure:

- `alphaquant.gptq` - GPTQ quantization
- `alphaquant.quantizers` - Quantizers (MXFP4, INT)
- `alphaquant.utils` - Utilities and helpers
- `lm_eval` - Language model evaluation
- `transformers` - Model loading
- `datasets` - Dataset loading
- `pandas` - Results analysis

## Notes

1. **Memory Requirements**: Each experiment loads the full model, so ensure sufficient GPU memory
2. **Time Estimates**: 
   - Baseline: ~10-15 minutes
   - MXFP4: ~15-20 minutes
   - GPTQ (each): ~30-45 minutes
   - Total pipeline: ~2-3 hours

3. **Reproducibility**: All scripts use fixed random seeds and save quantization plans

4. **Modularity**: Each script can run independently, making it easy to:
   - Re-run specific experiments
   - Try different hyperparameters
   - Add new quantization methods

## Extending the Pipeline

### Add New Quantization Method

1. Create new script (e.g., `quantize_custom.py`)
2. Follow the pattern from existing scripts
3. Add to `run_pipeline.py`
4. Update analysis script if needed

### Add New Calibration Dataset

1. Add loader to `data_utils.py`
2. Update `quantize_gptq.py` to support new dataset
3. Add option to pipeline

### Customize Analysis

Edit `analyze_results.py` to add:
- Additional metrics
- Visualizations
- Statistical tests
- Cost/benefit analysis

## Troubleshooting

**Issue**: Out of memory during GPTQ
- Solution: Reduce `--num_calibration_samples` or use smaller `--seqlen`

**Issue**: Slow evaluation
- Solution: Increase `--batch_size` (if memory allows)

**Issue**: Import errors
- Solution: Ensure you're running from project root and dependencies are installed

**Issue**: Different results across runs
- Solution: Check random seed consistency in data loading

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{alphaquant_gsm8k_analysis,
  title = {GSM8K Quantization Analysis Pipeline},
  author = {AlphaQuant Team},
  year = {2024},
  url = {https://github.com/your-repo/AlphaQuant}
}
```

