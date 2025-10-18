# 运行脚本使用指南

本目录包含5个独立的运行脚本，对应AlphaQuant的5个主要步骤。你可以单独运行每个步骤，也可以按顺序运行完整流程。

## 📝 脚本列表

| 脚本 | 功能 | 输入 | 输出 |
|------|------|------|------|
| `run_1_compute_alpha.sh` | 计算Alpha-Hill值 | 模型 | `results/alpha_values.csv` |
| `run_2_allocate_bitwidth.sh` | 分配量化位宽 | Alpha CSV | `configs/auto_quant_config.json` |
| `run_3_gptq_quantize.sh` | GPTQ权重量化 | 模型+配置 | `results/quantized_model.pt` |
| `run_4_evaluate_model.sh` | 评估量化模型 | 量化模型 | `results/eval_results.json` |
| `run_5_analyze_results.sh` | 分析和可视化 | Alpha CSV | `results/analysis.png` |

## 🚀 快速开始

### 方式1: 完整流程（使用原有脚本）

```bash
./run_pipeline.sh allenai/OLMoE-1B-7B-0924 cuda 0.3
```

### 方式2: 逐步运行（使用新的独立脚本）

```bash
# 步骤1: 计算Alpha值
./run_1_compute_alpha.sh allenai/OLMoE-1B-7B-0924 cuda

# 步骤2: 分配位宽
./run_2_allocate_bitwidth.sh allenai/OLMoE-1B-7B-0924 0.3

# 步骤3: GPTQ量化
./run_3_gptq_quantize.sh allenai/OLMoE-1B-7B-0924 cuda

# 步骤4: 评估模型
./run_4_evaluate_model.sh allenai/OLMoE-1B-7B-0924 cuda

# 步骤5: 分析结果
./run_5_analyze_results.sh visualize
```

## 📖 详细使用说明

### 步骤1: 计算Alpha-Hill值

```bash
./run_1_compute_alpha.sh <model> <device> [output]

# 参数:
#   model  - 模型名称或路径（默认: allenai/OLMoE-1B-7B-0924）
#   device - 设备 cuda/cpu（默认: cuda）
#   output - 输出文件路径（默认: results/alpha_values.csv）

# 示例:
./run_1_compute_alpha.sh meta-llama/Llama-2-7b-hf cpu
./run_1_compute_alpha.sh allenai/OLMoE-1B-7B-0924 cuda results/my_alpha.csv
```

**输出**: CSV文件，包含每层的Alpha-Hill敏感度值

---

### 步骤2: 分配量化位宽

```bash
./run_2_allocate_bitwidth.sh <model> <mxfp4_ratio> [bf16_ratio] [alpha_csv] [output]

# 参数:
#   model       - 模型名称或路径（默认: allenai/OLMoE-1B-7B-0924）
#   mxfp4_ratio - MXFP4高精度比例 0-1（默认: 0.3）
#   bf16_ratio  - BF16保留比例 0-1（默认: 0.0）
#   alpha_csv   - Alpha值CSV文件（默认: results/alpha_values.csv）
#   output      - 输出配置文件（默认: configs/auto_quant_config.json）

# 示例:
./run_2_allocate_bitwidth.sh allenai/OLMoE-1B-7B-0924 0.5
./run_2_allocate_bitwidth.sh meta-llama/Llama-2-7b-hf 0.3 0.1
```

**注意**: 
- 默认跳过attention层和routing gate/router层
- 混合精度只应用于gate_proj/up_proj/down_proj
- mxfp4_ratio=0.3 表示30%最敏感的层使用MXFP4，其余用MXFP8

**输出**: JSON配置文件，包含层级量化设置

---

### 步骤3: GPTQ权重量化

```bash
./run_3_gptq_quantize.sh <model> <device> [dataset] [nsamples] [config] [output]

# 参数:
#   model    - 模型名称或路径（默认: allenai/OLMoE-1B-7B-0924）
#   device   - 设备 cuda/cpu（默认: cuda）
#   dataset  - 校准数据集（默认: wikitext2，可选: c4, ptb）
#   nsamples - 校准样本数（默认: 128）
#   config   - 量化配置文件（默认: configs/auto_quant_config.json）
#   output   - 输出模型文件（默认: results/quantized_model.pt）

# 示例:
./run_3_gptq_quantize.sh allenai/OLMoE-1B-7B-0924 cuda wikitext2 128
./run_3_gptq_quantize.sh meta-llama/Llama-2-7b-hf cuda c4 256
```

**输出**: 量化后的模型权重文件

---

### 步骤4: 评估量化模型

```bash
./run_4_evaluate_model.sh <model> <device> [tasks] [batch_size] [checkpoint] [output]

# 参数:
#   model      - 原始模型名称或路径（默认: allenai/OLMoE-1B-7B-0924）
#   device     - 设备 cuda/cpu（默认: cuda）
#   tasks      - 评估任务，逗号分隔（默认: hellaswag,arc_easy,winogrande）
#   batch_size - 批次大小（默认: 8）
#   checkpoint - 量化模型文件（默认: results/quantized_model.pt）
#   output     - 输出结果文件（默认: results/eval_results.json）

# 示例:
./run_4_evaluate_model.sh allenai/OLMoE-1B-7B-0924 cuda "mmlu,gsm8k" 4
./run_4_evaluate_model.sh meta-llama/Llama-2-7b-hf cuda "hellaswag,arc_easy" 8
```

**支持的任务**: MMLU, HellaSwag, ARC-Easy, ARC-Challenge, WinoGrande, GSM8K等

**输出**: JSON文件，包含各任务的评估指标

---

### 步骤5: 分析和可视化结果

```bash
./run_5_analyze_results.sh <mode> [alpha_csv] [output] [model]

# 模式:
#   visualize - 可视化Alpha分布（默认）
#   alpha_mse - 分析Alpha-MSE关系（需要model参数）
#   compare   - 比较评估结果

# 参数:
#   mode      - 分析模式（默认: visualize）
#   alpha_csv - Alpha值CSV文件（默认: results/alpha_values.csv）
#   output    - 输出图片文件（默认: results/alpha_distribution.png）
#   model     - 模型路径（仅alpha_mse模式需要）

# 示例:
./run_5_analyze_results.sh visualize
./run_5_analyze_results.sh alpha_mse results/alpha_values.csv results/analysis.png allenai/OLMoE-1B-7B-0924
```

**输出**: 分析图表（PNG格式）

---

## 💡 使用场景

### 场景1: 完整流程

```bash
# 使用默认参数运行完整流程
./run_1_compute_alpha.sh
./run_2_allocate_bitwidth.sh
./run_3_gptq_quantize.sh
./run_4_evaluate_model.sh
./run_5_analyze_results.sh
```

### 场景2: 只做Alpha分析

```bash
# 步骤1和5
./run_1_compute_alpha.sh allenai/OLMoE-1B-7B-0924 cuda
./run_5_analyze_results.sh visualize results/alpha_values.csv results/alpha_viz.png
```

### 场景3: 跳过某些步骤

```bash
# 如果已有Alpha值，从步骤2开始
./run_2_allocate_bitwidth.sh allenai/OLMoE-1B-7B-0924 0.4
./run_3_gptq_quantize.sh allenai/OLMoE-1B-7B-0924 cuda
./run_4_evaluate_model.sh allenai/OLMoE-1B-7B-0924 cuda
```

### 场景4: 试验不同的mxfp4比例

```bash
# 使用同一个Alpha值，尝试不同配置
./run_1_compute_alpha.sh allenai/OLMoE-1B-7B-0924 cuda

# 配置1: 30% MXFP4
./run_2_allocate_bitwidth.sh allenai/OLMoE-1B-7B-0924 0.3 0.0 results/alpha_values.csv configs/config_30.json
./run_3_gptq_quantize.sh allenai/OLMoE-1B-7B-0924 cuda wikitext2 128 configs/config_30.json results/model_30.pt

# 配置2: 50% MXFP4
./run_2_allocate_bitwidth.sh allenai/OLMoE-1B-7B-0924 0.5 0.0 results/alpha_values.csv configs/config_50.json
./run_3_gptq_quantize.sh allenai/OLMoE-1B-7B-0924 cuda wikitext2 128 configs/config_50.json results/model_50.pt

# 评估对比
./run_4_evaluate_model.sh allenai/OLMoE-1B-7B-0924 cuda "mmlu,gsm8k" 8 results/model_30.pt results/eval_30.json
./run_4_evaluate_model.sh allenai/OLMoE-1B-7B-0924 cuda "mmlu,gsm8k" 8 results/model_50.pt results/eval_50.json
```

## 🔧 故障排除

### 权限问题

如果遇到权限错误，运行：
```bash
chmod +x run_*.sh
```

### GPU内存不足

使用CPU或减小batch size：
```bash
./run_1_compute_alpha.sh model cpu
./run_4_evaluate_model.sh model cuda "task1" 1
```

### 文件不存在

确保按顺序运行脚本，每步的输出是下一步的输入。

## 📂 输出文件结构

运行完整流程后，文件结构如下：

```
AlphaQuant/
├── results/
│   ├── alpha_values.csv           # 步骤1输出
│   ├── quantized_model.pt         # 步骤3输出
│   ├── eval_results.json          # 步骤4输出
│   └── alpha_distribution.png     # 步骤5输出
└── configs/
    └── auto_quant_config.json     # 步骤2输出
```

## 🆚 对比：独立脚本 vs 完整流程

| 特性 | `run_pipeline.sh` | 独立脚本 (`run_1_*.sh` ... `run_5_*.sh`) |
|------|-------------------|----------------------------------------|
| 运行方式 | 一键运行全部步骤 | 单独运行每个步骤 |
| 灵活性 | 低 | 高 |
| 参数控制 | 有限 | 完全控制 |
| 适用场景 | 快速试验 | 研究和调试 |
| 错误处理 | 遇错即停 | 可从任意步骤重启 |

## 📌 小贴士

1. **保存中间结果**: 默认文件名会被覆盖，建议使用自定义输出路径
2. **GPU显存**: 大模型建议使用`nsamples=32`减少内存占用
3. **快速试验**: 步骤1和2可以快速运行，用于调整配置
4. **评估耗时**: 步骤4最耗时，可先用少量任务测试
5. **可视化**: 步骤5可以多次运行，尝试不同的分析模式

---

**更多信息请查看**: `README.md` 和 `QUICKSTART.md`

