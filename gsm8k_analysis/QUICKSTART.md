# GSM8K Analysis - Quick Start Guide

## 一分钟快速开始

### 最简单的运行方式

```bash
cd /Users/superone77/Code/AlphaQuant
bash gsm8k_analysis/run_pipeline.sh
```

就这么简单！Pipeline 会自动完成所有步骤。

## 详细步骤说明

### Pipeline 包含的实验

1. **Baseline**: 评估原始 OLMoE 模型
2. **MXFP4**: 所有 expert 层量化到 MXFP4
3. **GPTQ + WikiText2**: 使用 WikiText2 作为校准数据
4. **GPTQ + GSM8K**: 使用 GSM8K 训练集作为校准数据

### 自定义参数

```bash
# 修改模型、样本数等参数
bash gsm8k_analysis/run_pipeline.sh \
    allenai/OLMoE-1B-7B-0924 \  # 模型名称
    256 \                        # GSM8K 测试样本数
    128 \                        # 校准样本数
    cuda \                       # 设备
    bfloat16 \                   # 数据类型
    1                            # batch size
```

### 只运行部分实验

```bash
# 跳过某些步骤
python gsm8k_analysis/run_pipeline.py \
    --skip_baseline \           # 跳过 baseline
    --skip_mxfp4 \              # 跳过 MXFP4
    --num_samples 128           # 使用更少样本加快测试
```

### 单独运行某个实验

```bash
# 只运行 GPTQ with GSM8K
python gsm8k_analysis/quantize_gptq.py \
    --model allenai/OLMoE-1B-7B-0924 \
    --calibration_data gsm8k \
    --num_calibration_samples 128 \
    --num_eval_samples 256
```

## 查看结果

### 结果文件位置

```
gsm8k_analysis/results/
├── baseline_20241019_143022.json          # Baseline 结果
├── mxfp4_20241019_143022.json             # MXFP4 结果
├── gptq_wikitext2_20241019_143022.json    # GPTQ+WikiText2 结果
├── gptq_gsm8k_20241019_143022.json        # GPTQ+GSM8K 结果
├── pipeline_summary_20241019_143022.json  # 总结
├── comparison_20241019_143022.csv         # 对比表格（CSV）
└── comparison_20241019_143022.txt         # 详细分析（文本）
```

### 快速查看对比结果

```bash
# 查看 CSV 表格
cat gsm8k_analysis/results/comparison_*.csv

# 查看详细分析
cat gsm8k_analysis/results/comparison_*.txt
```

### 使用 Python 查看结果

```python
import json
import pandas as pd

# 读取对比表格
df = pd.read_csv('gsm8k_analysis/results/comparison_20241019_143022.csv')
print(df)

# 读取某个实验的详细结果
with open('gsm8k_analysis/results/gptq_gsm8k_20241019_143022.json') as f:
    results = json.load(f)
    print(results['results']['gsm8k'])
```

## 预期运行时间

基于单个 GPU (A100/H100):

- Baseline 评估: ~10-15 分钟
- MXFP4 量化: ~15-20 分钟  
- GPTQ + WikiText2: ~30-45 分钟
- GPTQ + GSM8K: ~30-45 分钟
- **总计**: ~2-3 小时

## 常见问题

### Q: 内存不足怎么办？

**A**: 减少样本数或使用更小的模型
```bash
python gsm8k_analysis/run_pipeline.py \
    --num_samples 128 \
    --num_calibration_samples 64
```

### Q: 如何只测试特定的量化方法？

**A**: 使用 skip 参数
```bash
python gsm8k_analysis/run_pipeline.py \
    --skip_baseline \
    --skip_mxfp4 \
    --skip_gptq_wikitext  # 只运行 GPTQ with GSM8K
```

### Q: 如何使用不同的模型？

**A**: 修改 --model 参数
```bash
python gsm8k_analysis/run_pipeline.py \
    --model allenai/OLMoE-1B-7B-0125-Instruct
```

### Q: 结果保存在哪里？

**A**: 所有结果保存在 `gsm8k_analysis/results/` 目录下，带时间戳

### Q: 如何重新分析已有结果？

**A**: 直接运行分析脚本
```bash
python gsm8k_analysis/analyze_results.py \
    --summary gsm8k_analysis/results/pipeline_summary_<timestamp>.json
```

## 进阶使用

### 修改量化参数

编辑 `quantize_gptq.py` 中的默认参数，或通过命令行指定：

```bash
python gsm8k_analysis/quantize_gptq.py \
    --calibration_data gsm8k \
    --bits 3 \                    # 使用 3-bit 量化
    --percdamp 0.02 \             # 修改 damping
    --blocksize 256 \             # 修改 block size
    --actorder                    # 启用 activation order
```

### 添加新的校准数据集

1. 在 `data_utils.py` 中添加新的数据加载函数
2. 在 `quantize_gptq.py` 中注册新数据集
3. 运行 pipeline

### 批量实验

创建脚本运行多组实验：

```bash
#!/bin/bash
# 测试不同样本数
for samples in 64 128 256 512; do
    python gsm8k_analysis/run_pipeline.py \
        --num_samples $samples \
        --output_dir "gsm8k_analysis/results/samples_$samples"
done
```

## 输出示例

成功运行后，你会看到类似这样的对比结果：

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
```

## 关键发现

基于我们的实验，你可能会发现：

1. **任务特定校准很重要**: 使用 GSM8K 校准比 WikiText2 效果好
2. **GPTQ 优于简单量化**: GPTQ 比 MXFP4 精度损失小
3. **量化对数学推理影响明显**: GSM8K 对量化比较敏感

## 下一步

- 尝试不同的 bits (2, 3, 8)
- 测试更大的模型
- 添加更多校准数据集
- 分析每层的量化敏感度
- 实现混合精度量化

## 获取帮助

```bash
# 查看各个脚本的帮助
python gsm8k_analysis/run_pipeline.py --help
python gsm8k_analysis/quantize_gptq.py --help
python gsm8k_analysis/analyze_results.py --help
```

