# Scripts Directory

This directory contains the core scripts and utilities for the AlphaQuant pipeline.

## Core Scripts (Used by Main Pipeline)

These scripts are integrated into the main pipeline (`run_pipeline.sh`):

- **`alpha_hill_quantization.py`** - Compute Alpha-Hill values and generate quantization configs
- **`run_gptq.py`** - GPTQ quantization with mixed precision support
- **`eval_with_plans.py`** - Evaluate models with quantization plans
- **`quantize_model.py`** - General quantization tool with layer-wise config

## Analysis & Visualization Scripts

- **`analyze_quantization_alpha_hill.py`** - Analyze Alpha-Hill across different quantization formats
- **`analyze_alpha_mse_relationship.py`** - Study relationship between alpha and MSE
- **`plot_alpha_mse.py`** - Plot alpha vs MSE graphs
- **`plot_alpha_vs_mse_reg_per_bit.py`** - Regression analysis for different bit widths
- **`visualize_quantization_alpha.py`** - Visualize alpha distribution and quantization effects
- **`calculate_avg_bits.py`** - Calculate average bits for mixed-precision configs

## Archived Scripts

The `archive/` directory contains:
- Test scripts (`test_*.py`) - Unit tests for specific features
- Example scripts (`example_*.py`) - Usage examples
- Experimental scripts - Research and development code
- Old documentation (`README_*.md`) - Superseded by main docs

## Usage

For the standard workflow, use the main pipeline scripts in the root directory:

```bash
# Full pipeline
./run_pipeline.sh allenai/OLMoE-1B-7B-0924 cuda 0.3

# Or run individual steps
python 1_compute_alpha.py --model <model> --output results/alpha.csv
python 2_allocate_bitwidth.py --alpha-csv results/alpha.csv --output configs/config.json
python 3_gptq_quantize.py --model <model> --config configs/config.json --save model.pt
python 4_evaluate_model.py --model <model> --checkpoint model.pt --output results.json
python 5_analyze_results.py --mode visualize --alpha-csv results/alpha.csv --output plot.png
```

For advanced usage and specific analysis, use the individual scripts in this directory.

