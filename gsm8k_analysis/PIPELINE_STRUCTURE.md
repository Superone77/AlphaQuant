# GSM8K Analysis Pipeline - Step-by-Step Structure

## 📋 概述

这个 pipeline 已经按照主项目的风格拆分成独立的步骤，每个步骤都可以单独运行。

## 🔧 关于 GPTQ 实现的说明

**重要**: 当前的 GPTQ 脚本使用标准 GPTQ 实现，但 AlphaQuant 有专门的 MoE 优化版本。

### MoE GPTQ vs 标准 GPTQ

AlphaQuant 提供了两种 GPTQ 实现：

1. **标准 GPTQ** (`alphaquant/gptq/gptq.py`):
   - 适用于所有模型
   - 使用标准 Hessian 计算
   - **当前使用的版本**

2. **MoE GPTQ** (`alphaquant/gptq/gptq_moe.py`):
   - 专门为 MoE 模型优化
   - 使用 routing-score-weighted Hessian
   - 跟踪 expert utilization
   - 支持 OLMoE 的 top-8 routing
   
### 如何切换到 MoE GPTQ

要使用 MoE 优化版本，需要修改步骤 3 和 4 的脚本：

```python
# 在 3_gptq_quantize_wikitext2.py 和 4_gptq_quantize_gsm8k.py 中

# 替换这一行：
from alphaquant.gptq.quantize import gptq_quantize_model

# 为：
from alphaquant.gptq.gptq_moe import create_gptq_for_layer, detect_moe_architecture
```

然后在量化时使用 `create_gptq_for_layer` 为每个 expert 层创建 `GPTQMoE` 实例。

**注意**: MoE GPTQ 需要捕获 routing 信息，实现更复杂但精度可能更好。

## 📁 文件结构

```
gsm8k_analysis/
├── # 核心 Python 脚本（按步骤）
├── 1_eval_baseline.py              # 步骤1: Baseline评估
├── 2_quantize_mxfp4_rtn.py         # 步骤2: MXFP4 (RTN)量化
├── 3_gptq_quantize_wikitext2.py    # 步骤3: GPTQ + WikiText2
├── 4_gptq_quantize_gsm8k.py        # 步骤4: GPTQ + GSM8K
├── 5_analyze_results.py            # 步骤5: 结果分析
│
├── # Shell 运行脚本（对应每个步骤）
├── run_1_eval_baseline.sh
├── run_2_quantize_mxfp4_rtn.sh
├── run_3_gptq_quantize_wikitext2.sh
├── run_4_gptq_quantize_gsm8k.sh
├── run_5_analyze_results.sh
├── run_all_steps.sh                # 运行完整 pipeline
│
├── # 原始整合版本（保留）
├── eval_baseline.py
├── quantize_mxfp4.py
├── quantize_gptq.py
├── run_pipeline.py
├── run_pipeline.sh
├── analyze_results.py
│
├── # 工具和文档
├── data_utils.py                   # GSM8K 数据加载
├── example_usage.py
├── test_setup.py
├── __init__.py
├── README.md
├── QUICKSTART.md
├── SUMMARY.md
└── PIPELINE_STRUCTURE.md           # 本文件
```

## 🚀 使用方法

### 方法 1: 运行完整 Pipeline

```bash
# 一键运行所有步骤
cd /Users/superone77/Code/AlphaQuant
./gsm8k_analysis/run_all_steps.sh

# 或自定义参数
./gsm8k_analysis/run_all_steps.sh \
    allenai/OLMoE-1B-7B-0924 \  # 模型
    256 \                        # 评估样本数
    128 \                        # 校准样本数
    cuda                         # 设备
```

### 方法 2: 单独运行各个步骤

```bash
# Step 1: Baseline 评估
./gsm8k_analysis/run_1_eval_baseline.sh

# Step 2: MXFP4 (RTN) 量化
./gsm8k_analysis/run_2_quantize_mxfp4_rtn.sh

# Step 3: GPTQ + MXFP4 (WikiText2 校准)
./gsm8k_analysis/run_3_gptq_quantize_wikitext2.sh

# Step 4: GPTQ + MXFP4 (GSM8K 校准)
./gsm8k_analysis/run_4_gptq_quantize_gsm8k.sh

# Step 5: 分析结果
./gsm8k_analysis/run_5_analyze_results.sh
```

### 方法 3: 直接运行 Python 脚本

```bash
# Step 1
python gsm8k_analysis/1_eval_baseline.py \
    --model allenai/OLMoE-1B-7B-0924 \
    --num_samples 256 \
    --device cuda

# Step 2
python gsm8k_analysis/2_quantize_mxfp4_rtn.py \
    --model allenai/OLMoE-1B-7B-0924 \
    --num_samples 256

# Step 3
python gsm8k_analysis/3_gptq_quantize_wikitext2.py \
    --model allenai/OLMoE-1B-7B-0924 \
    --num_calibration_samples 128 \
    --num_eval_samples 256

# Step 4
python gsm8k_analysis/4_gptq_quantize_gsm8k.py \
    --model allenai/OLMoE-1B-7B-0924 \
    --num_calibration_samples 128 \
    --num_eval_samples 256

# Step 5
python gsm8k_analysis/5_analyze_results.py
```

## 📊 Pipeline 步骤详解

### Step 1: Baseline 评估
- **脚本**: `1_eval_baseline.py`
- **功能**: 评估原始（未量化）OLMoE 模型
- **输出**: `results/baseline.json`
- **时间**: ~10-15 分钟

### Step 2: MXFP4 (RTN) 量化
- **脚本**: `2_quantize_mxfp4_rtn.py`
- **功能**: 使用 RTN（无校准）量化所有 expert 层到 MXFP4
- **输出**: 
  - `results/mxfp4_rtn.json`
  - `configs/mxfp4_rtn_plan.json`
- **时间**: ~15-20 分钟

### Step 3: GPTQ + MXFP4 (WikiText2)
- **脚本**: `3_gptq_quantize_wikitext2.py`
- **功能**: 使用 WikiText2 校准数据进行 GPTQ 量化
- **校准**: 通用文本数据
- **输出**:
  - `results/gptq_mxfp4_wikitext2.json`
  - `configs/gptq_mxfp4_wikitext2_plan.json`
- **时间**: ~30-45 分钟

### Step 4: GPTQ + MXFP4 (GSM8K)
- **脚本**: `4_gptq_quantize_gsm8k.py`
- **功能**: 使用 GSM8K 训练集校准数据进行 GPTQ 量化
- **校准**: 任务特定数据
- **输出**:
  - `results/gptq_mxfp4_gsm8k.json`
  - `configs/gptq_mxfp4_gsm8k_plan.json`
- **时间**: ~30-45 分钟

### Step 5: 结果分析
- **脚本**: `5_analyze_results.py`
- **功能**: 对比所有实验结果，生成分析报告
- **输出**:
  - `results/comparison.csv` - 对比表格
  - `results/comparison.txt` - 详细分析
- **时间**: <1 分钟

## 🎯 与主项目的对应关系

| GSM8K Analysis | 主项目 | 说明 |
|---------------|--------|------|
| `1_eval_baseline.py` | - | 新增（评估基准） |
| `2_quantize_mxfp4_rtn.py` | - | 新增（RTN 量化） |
| `3_gptq_quantize_wikitext2.py` | `3_gptq_quantize.py` | 类似（GPTQ 量化） |
| `4_gptq_quantize_gsm8k.py` | `3_gptq_quantize.py` | 类似（不同校准数据） |
| `5_analyze_results.py` | `5_analyze_results.py` | 类似（结果分析） |

## 🔄 原始 vs 拆分版本

### 原始版本（保留）
- `run_pipeline.py` - 整合所有步骤的 Python 脚本
- `run_pipeline.sh` - 整合所有步骤的 Shell 脚本
- 优点：一次性运行，简单
- 缺点：难以单独调试某个步骤

### 拆分版本（新增）
- `1_*.py`, `2_*.py`, ..., `5_*.py` - 独立的步骤脚本
- `run_1_*.sh`, `run_2_*.sh`, ..., `run_5_*.sh` - 对应的运行脚本
- `run_all_steps.sh` - 按顺序运行所有步骤
- 优点：
  - 每步独立，便于调试
  - 可以跳过已完成的步骤
  - 与主项目风格一致
- 缺点：文件较多

## 💡 最佳实践

### 1. 首次运行
```bash
# 使用少量样本快速测试
./gsm8k_analysis/run_all_steps.sh allenai/OLMoE-1B-7B-0924 64 32 cuda
```

### 2. 完整实验
```bash
# 使用完整样本数
./gsm8k_analysis/run_all_steps.sh allenai/OLMoE-1B-7B-0924 256 128 cuda
```

### 3. 重跑某个步骤
```bash
# 如果步骤 3 失败，单独重跑
./gsm8k_analysis/run_3_gptq_quantize_wikitext2.sh
```

### 4. 调试模式
```bash
# 直接运行 Python 脚本，可以看到详细输出
python gsm8k_analysis/3_gptq_quantize_wikitext2.py \
    --num_calibration_samples 32 \
    --num_eval_samples 64
```

## 📝 输出文件

### 结果文件
```
results/
├── baseline.json                    # Baseline 结果
├── mxfp4_rtn.json                   # MXFP4 (RTN) 结果
├── gptq_mxfp4_wikitext2.json       # GPTQ + WikiText2 结果
├── gptq_mxfp4_gsm8k.json           # GPTQ + GSM8K 结果
├── comparison.csv                   # 对比表格
└── comparison.txt                   # 详细分析
```

### 配置文件
```
configs/
├── mxfp4_rtn_plan.json             # MXFP4 量化计划
├── gptq_mxfp4_wikitext2_plan.json  # GPTQ WikiText2 计划
└── gptq_mxfp4_gsm8k_plan.json      # GPTQ GSM8K 计划
```

## 🔍 故障排查

### 内存不足
```bash
# 减少样本数
python gsm8k_analysis/1_eval_baseline.py --num_samples 64
```

### GPTQ 太慢
```bash
# 减少校准样本
python gsm8k_analysis/3_gptq_quantize_wikitext2.py \
    --num_calibration_samples 64
```

### 查看某个步骤的帮助
```bash
python gsm8k_analysis/1_eval_baseline.py --help
```

## 📚 扩展阅读

- `README.md` - 完整文档
- `QUICKSTART.md` - 快速开始指南
- `SUMMARY.md` - 项目总结
- `test_setup.py` - 环境测试

## ✅ 检查清单

运行 pipeline 前确认：

- [ ] GPU 内存充足（建议 40GB+）
- [ ] 安装了所有依赖
- [ ] 有足够的磁盘空间（~20GB）
- [ ] 网络连接正常（下载数据集和模型）

运行后检查：

- [ ] 所有 5 个步骤都成功完成
- [ ] `results/` 目录包含所有结果文件
- [ ] `comparison.csv` 显示准确率对比
- [ ] `comparison.txt` 包含详细分析

