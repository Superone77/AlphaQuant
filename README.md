# AlphaQuant

**Alpha-guided Mixed-Precision Quantization for Large Language Models**

AlphaQuant is a complete quantization framework that uses Alpha-Hill sensitivity metrics to automatically determine optimal bit-width allocation for different layers in LLMs. It supports various quantization formats and includes GPTQ-based weight optimization.

## 🎯 Key Features

- ✅ **Alpha-Hill Metric**: Quantization sensitivity measurement for layer-wise analysis
- ✅ **Automatic Bitwidth Allocation**: Data-driven precision assignment based on sensitivity
- ✅ **GPTQ Weight Optimization**: Hessian-based post-training quantization
- ✅ **Hadamard Transform**: Outlier suppression for better low-bit quantization *(NEW!)*
- ✅ **Mixed-Precision Support**: INT2/3/4/6/8, FP4/6/8, MXFP4/6/8
- ✅ **MoE Model Support**: OLMoE, Mixtral, Qwen-MoE, DeepSeek-MoE
- ✅ **Complete Pipeline**: From alpha computation to model evaluation

## 🚀 Quick Start

### Complete Pipeline

Run the entire quantization pipeline with one command:

```bash
./run_pipeline.sh <model_name> <device> <mxfp4_ratio>

# Example
./run_pipeline.sh allenai/OLMoE-1B-7B-0924 cuda 0.3
```

This will:
1. Compute Alpha-Hill values for all layers
2. Allocate bitwidth based on sensitivity (30% high-precision mxfp4, 70% mxfp8)
3. Apply GPTQ quantization
3.5. (Optional) Finetune router weights only
4. Evaluate the quantized model
5. Generate analysis visualizations

### Step-by-Step Usage

#### Step 1: Compute Alpha-Hill Values

```bash
python 1_compute_alpha.py \
    --model allenai/OLMoE-1B-7B-0924 \
    --output results/alpha_values.csv \
    --device cuda
```

**Output**: CSV file with Alpha-Hill sensitivity values for each layer.

#### Step 2: Allocate Bitwidth

```bash
python 2_allocate_bitwidth.py \
    --model allenai/OLMoE-1B-7B-0924 \
    --alpha-csv results/alpha_values.csv \
    --mxfp4-ratio 0.3 \
    --output configs/auto_quant_config.json
```

**Output**: JSON configuration with layer-wise quantization settings.

**Note**: By default, attention layers (q_proj, k_proj, v_proj, o_proj) and **routing gate/router** layers are NOT quantized as they are critical for model performance. Mixed precision allocation is applied to remaining layers (mainly **gate_proj, up_proj, down_proj** in MoE FFN blocks). Use `--no-skip-attention` or `--no-skip-gate` to override this behavior.

#### Step 3: GPTQ Quantization

```bash
python 3_gptq_quantize.py \
    --model allenai/OLMoE-1B-7B-0924 \
    --config configs/auto_quant_config.json \
    --dataset wikitext2 \
    --nsamples 128 \
    --use-hadamard \
    --save results/quantized_model.pt
```

**Output**: Quantized model checkpoint with optimized weights.

**New Feature - Hadamard Transform:**
- Add `--use-hadamard` flag to enable outlier suppression
- Redistributes activation outliers for better quantization
- Especially effective for low-bit (2-4 bit) quantization
- No runtime overhead (fused into weights during quantization)
- See [Hadamard Transform Guide](docs/HADAMARD_TRANSFORM.md) for details

#### Step 3.5: Router Finetuning (Optional)

```bash
python 3.5_router_finetuning.py \
    --model allenai/OLMoE-1B-7B-0924 \
    --checkpoint results/quantized_model.pt \
    --lr 1e-4 \
    --batch_size 1 \
    --weight_decay 1e-4 \
    --num_epochs 1 \
    --save results/router_finetuned_model.pt
```

**Output**: Model with finetuned router weights. This step freezes all attention and expert weights, and only finetunes the router/gate layers to minimize cross-entropy loss on the calibration dataset (128 sequences of length 2048 from WikiText2).

#### Step 4: Evaluate Model

```bash
python 4_evaluate_model.py \
    --model allenai/OLMoE-1B-7B-0924 \
    --checkpoint results/quantized_model.pt \
  --tasks hellaswag,arc_easy,winogrande \
    --output results/eval_results.json
```

**Output**: Evaluation results on downstream tasks.

#### Step 5: Analyze Results

```bash
# Visualize alpha distribution
python 5_analyze_results.py \
    --mode visualize \
    --alpha-csv results/alpha_values.csv \
    --output results/alpha_distribution.png

# Analyze alpha-MSE relationship
python 5_analyze_results.py \
    --mode alpha_mse \
    --model allenai/OLMoE-1B-7B-0924 \
    --alpha-csv results/alpha_values.csv \
    --output results/alpha_mse_analysis.png
```

**Output**: Analysis plots and statistics.

## 📁 Project Structure

```
AlphaQuant/
├── 1_compute_alpha.py          # Step 1: Compute Alpha-Hill values
├── 2_allocate_bitwidth.py      # Step 2: Automatic bitwidth allocation
├── 3_gptq_quantize.py          # Step 3: GPTQ weight quantization
├── 3.5_router_finetuning.py    # Step 3.5: Router finetuning (optional)
├── 4_evaluate_model.py         # Step 4: Model evaluation
├── 5_analyze_results.py        # Step 5: Results analysis
├── run_pipeline.sh             # Complete pipeline runner
│
├── alphaquant/                 # Core library
│   ├── alpha_hill/             # Alpha-Hill computation
│   ├── gptq/                   # GPTQ quantization
│   ├── quantizers/             # Quantizer implementations
│   ├── modules/                # Custom modules
│   └── utils/                  # Utilities
│
├── models/                     # Independent model implementations
│   ├── olmoe/                  # OLMoE model
│   ├── qwen_moe_14b_chat/      # Qwen2-MoE model
│   ├── mixtral_model/          # Mixtral model
│   └── deepseek_moe_16b_chat/  # DeepSeek-MoE model
│
├── configs/                    # Quantization configurations
├── scripts/                    # Additional analysis scripts
├── docs/                       # Documentation
├── examples/                   # Usage examples
├── results/                    # Output directory (auto-created)
└── README.md                   # This file
```

## 📊 Main Functionality

### 1. Alpha-Hill Computation

The Alpha-Hill metric measures layer sensitivity to quantization:

```python
from alphaquant.alpha_hill.utils import alpha_hill_from_model
from alphaquant.utils.hf_utils import load_hf_causal_lm

model = load_hf_causal_lm("meta-llama/Llama-2-7b-hf")
alpha_values = alpha_hill_from_model(model)
```

**Higher alpha = More sensitive to quantization = Needs higher precision**

### 2. Automatic Bitwidth Allocation

Based on alpha values, automatically assign precision:
- Top N% sensitive layers → Keep in BF16 or use high-precision (MXFP4)
- Moderately sensitive → Medium precision (MXFP6/8)
- Less sensitive → Lower precision (INT4/8)

### 3. GPTQ Weight Quantization

Uses Hessian information to optimize quantized weights:
- Minimizes layer-wise reconstruction error
- Supports mixed-precision configurations
- Special handling for MoE expert layers

### 4. Model Evaluation

Evaluate on standard benchmarks:
- MMLU, HellaSwag, ARC, WinoGrande
- GSM8K for math reasoning
- Custom task support via lm-eval-harness

### 5. Data Analysis

- Alpha distribution visualization
- Alpha-MSE relationship analysis
- Layer-wise sensitivity comparison
- Quantization effect analysis

## 🔧 Advanced Usage

### Manual Configuration

Create a custom quantization config:

```json
{
  "default": {
    "wq": "mxfp8",
    "aq": "mxfp8",
    "group_size": 128
  },
  "overrides": [
    {
      "pattern": "model.layers.0.*",
      "wq": "mxfp4",
      "group_size": 64,
      "comment": "First layer - use high precision"
    },
    {
      "pattern": "*.experts.*",
      "wq": "mxfp6",
      "comment": "Expert layers - medium precision"
    }
  ]
}
```

### Using Core Scripts Directly

For more control, use the scripts in `scripts/` directory:

```bash
# Alpha-Hill quantization with custom ratios
python scripts/alpha_hill_quantization.py \
    --model <model> \
    --mxfp4-ratio 0.3 \
    --bf16-ratio 0.1 \
    --output-config configs/custom.json

# GPTQ with custom settings
python scripts/run_gptq.py \
    --model <model> \
    --config configs/custom.json \
    --dataset c4 \
    --groupsize 64 \
    --actorder

# Evaluation with quantization plan
python scripts/eval_with_plans.py \
    --model <model> \
    --plan configs/custom.json \
    --tasks gsm8k,mmlu
```

## 📖 Documentation

Comprehensive documentation is available in the `docs/` directory:

- **`docs/HADAMARD_TRANSFORM.md`** - **NEW!** Hadamard transform for outlier suppression
- **`docs/GPTQ_PIPELINE_README.md`** - GPTQ implementation details
- **`docs/GPTQ_QUICKSTART.md`** - Quick start guide for GPTQ
- **`docs/ROUTER_FINETUNING.md`** - Router-only finetuning guide
- **`docs/MODEL_INTEGRATION_SUMMARY.md`** - MoE model integration guide
- **`docs/GROUP_QUANTIZATION_README.md`** - Group quantization documentation
- **`models/README.md`** - Model usage and architecture details
- **`scripts/README.md`** - Script documentation

## 🔬 Supported Quantization Formats

| Format | Bits | Description | Use Case |
|--------|------|-------------|----------|
| **INT2-8** | 2-8 | Integer quantization | High compression |
| **FP4/6/8** | 4-8 | Floating point | Better dynamic range |
| **MXFP4/6/8** | 4-8 | Microscaling FP | Optimal accuracy/size |
| **BF16** | 16 | Keep original | Sensitive layers |

## 🎓 Supported Models

AlphaQuant includes independent model implementations (no transformers version dependency):

| Model | Path | Special Features |
|-------|------|------------------|
| **OLMoE** | `models/olmoe/` | Top-8 routing |
| **Qwen2-MoE** | `models/qwen_moe_14b_chat/` | Shared expert |
| **Mixtral** | `models/mixtral_model/` | Top-2 sparse MoE |
| **DeepSeek-MoE** | `models/deepseek_moe_16b_chat/` | Multi-level routing |

## 📦 Installation

```bash
# Clone repository
git clone <your-repo-url>
cd AlphaQuant

# Install dependencies
pip install -r requirements.txt

# Make scripts executable
chmod +x run_pipeline.sh
```

## 🧪 Example Results

Typical results on OLMoE-1B-7B model:

| Configuration | Avg Bits | MMLU | HellaSwag | Model Size |
|---------------|----------|------|-----------|------------|
| BF16 (baseline) | 16.0 | 0.XX | 0.XX | 100% |
| Alpha-based (30% MXFP4) | 6.4 | 0.XX | 0.XX | 40% |
| Uniform MXFP8 | 8.0 | 0.XX | 0.XX | 50% |

*Fill in actual results from your experiments*

## 🤝 Contributing

This is a research project. Feel free to:
- Report issues
- Suggest improvements
- Add new quantization methods
- Extend MoE model support

## 📄 License

[Add your license here]

## 🙏 Acknowledgments

- GPTQ: [IST-DASLab/gptq](https://github.com/IST-DASLab/gptq)
- MoEQuant: Quantization strategies for MoE models
- lm-evaluation-harness: Model evaluation framework

## 📧 Contact

[Add your contact information]

---

**For detailed workflow and technical documentation, see the `docs/` directory.**
