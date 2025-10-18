# AlphaQuant 快速开始指南

## 🎯 5分钟快速上手

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行完整流程

```bash
# 使用OLMoE模型示例
./run_pipeline.sh allenai/OLMoE-1B-7B-0924 cuda 0.3

# 参数说明：
# - 第1个参数: 模型名称或路径
# - 第2个参数: 设备 (cuda/cpu)
# - 第3个参数: MXFP4高精度比例 (0.0-1.0)
```

这将自动完成：
1. ✅ 计算Alpha-Hill敏感度值
2. ✅ 自动分配量化位宽
3. ✅ GPTQ权重优化
4. ✅ 模型评估
5. ✅ 结果分析

### 3. 查看结果

```bash
# Alpha值
cat results/alpha_values.csv

# 量化配置
cat configs/auto_quant_config.json

# 评估结果
cat results/eval_results.json

# 分析图表
open results/alpha_distribution.png
```

## 📝 分步运行（高级用法）

### 步骤1: 计算Alpha值

```bash
python 1_compute_alpha.py \
    --model allenai/OLMoE-1B-7B-0924 \
    --output results/alpha_values.csv \
    --device cuda
```

**作用**: 计算每层的量化敏感度
**输出**: `results/alpha_values.csv`

### 步骤2: 分配位宽

```bash
python 2_allocate_bitwidth.py \
    --model allenai/OLMoE-1B-7B-0924 \
    --alpha-csv results/alpha_values.csv \
    --mxfp4-ratio 0.3 \
    --bf16-ratio 0.0 \
    --output configs/auto_quant_config.json
```

**作用**: 基于敏感度自动分配精度
**输出**: `configs/auto_quant_config.json`

**参数说明**:
- `--mxfp4-ratio`: 使用高精度MXFP4的层比例（敏感层）
- `--bf16-ratio`: 保持BF16不量化的层比例（最敏感层）
- `--skip-attention`: 跳过attention层量化（默认开启）
- `--skip-gate`: 跳过gate/router层量化（默认开启）
- `--no-skip-attention`: 如果要量化attention层，使用此选项
- `--no-skip-gate`: 如果要量化gate/router层，使用此选项
- 剩余层使用MXFP8中精度

**⚠️ 重要**: 默认情况下，attention层（q_proj, k_proj, v_proj, o_proj）和gate/router层不会被量化，因为它们对模型性能影响较大。

### 步骤3: GPTQ量化

```bash
python 3_gptq_quantize.py \
    --model allenai/OLMoE-1B-7B-0924 \
    --config configs/auto_quant_config.json \
    --dataset wikitext2 \
    --nsamples 128 \
    --save results/quantized_model.pt \
    --device cuda
```

**作用**: 使用GPTQ算法优化量化权重
**输出**: `results/quantized_model.pt`

**可选参数**:
- `--dataset`: 校准数据集 (wikitext2/c4/ptb)
- `--nsamples`: 校准样本数
- `--groupsize`: GPTQ分组大小
- `--use-rtn`: 使用RTN替代GPTQ（更快但精度略低）

### 步骤4: 评估模型

```bash
python 4_evaluate_model.py \
    --model allenai/OLMoE-1B-7B-0924 \
    --checkpoint results/quantized_model.pt \
    --tasks hellaswag,arc_easy,winogrande \
    --batch_size 8 \
    --output results/eval_results.json \
    --device cuda
```

**作用**: 在下游任务上评估量化模型
**输出**: `results/eval_results.json`

**支持的任务**: 
- MMLU, HellaSwag, ARC-Easy, ARC-Challenge
- WinoGrande, GSM8K, 等

### 步骤5: 分析结果

```bash
# 可视化Alpha分布
python 5_analyze_results.py \
    --mode visualize \
    --alpha-csv results/alpha_values.csv \
    --output results/alpha_distribution.png

# 分析Alpha-MSE关系
python 5_analyze_results.py \
    --mode alpha_mse \
    --model allenai/OLMoE-1B-7B-0924 \
    --alpha-csv results/alpha_values.csv \
    --output results/alpha_mse_analysis.png
```

**作用**: 生成分析图表和统计信息
**输出**: 可视化图表

## 💡 常见使用场景

### 场景1: 快速评估量化效果

```bash
# 使用默认配置快速量化和评估
./run_pipeline.sh your-model cuda 0.3
```

### 场景2: 自定义位宽分配

```bash
# 步骤1-2: 生成自定义配置
python 1_compute_alpha.py --model your-model --output alpha.csv
python 2_allocate_bitwidth.py \
    --alpha-csv alpha.csv \
    --mxfp4-ratio 0.5 \
    --bf16-ratio 0.1 \
    --output custom_config.json

# 手动编辑 custom_config.json

# 步骤3-5: 使用自定义配置
python 3_gptq_quantize.py --config custom_config.json --save model.pt
python 4_evaluate_model.py --checkpoint model.pt --output results.json
```

### 场景3: 只做Alpha分析

```bash
python 1_compute_alpha.py --model your-model --output alpha.csv
python 5_analyze_results.py --mode visualize --alpha-csv alpha.csv --output plot.png
```

### 场景4: 使用已有的量化配置

```bash
# 跳过步骤1-2，直接使用预定义配置
python 3_gptq_quantize.py \
    --model your-model \
    --config configs/gptq_mixed_precision.json \
    --save model.pt
```

## 🔧 配置文件示例

手动创建量化配置 `configs/my_config.json`:

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
      "wq": "bf16",
      "skip": true,
      "comment": "第一层保持BF16"
    },
    {
      "pattern": "*.experts.*",
      "wq": "mxfp4",
      "group_size": 64,
      "comment": "专家层使用高精度"
    },
    {
      "pattern": "*.lm_head",
      "skip": true,
      "comment": "跳过输出层"
    }
  ]
}
```

然后使用：
```bash
python 3_gptq_quantize.py --config configs/my_config.json --save model.pt
```

## 📊 理解Alpha值

Alpha-Hill值表示层对量化的敏感度：

- **高Alpha (> 0.8)**: 非常敏感，建议BF16或MXFP4
- **中Alpha (0.5-0.8)**: 中等敏感，适合MXFP6/8
- **低Alpha (< 0.5)**: 不敏感，可以用INT4/8

查看Alpha分布：
```bash
python 5_analyze_results.py --mode visualize --alpha-csv results/alpha_values.csv
```

## ⚙️ 性能优化建议

### 快速试验（RTN模式）
```bash
python 3_gptq_quantize.py --use-rtn --nsamples 32 ...
```

### 高质量量化（GPTQ模式）
```bash
python 3_gptq_quantize.py --use-gptq --nsamples 256 --actorder ...
```

### GPU内存优化
```bash
# 使用CPU加载
python 1_compute_alpha.py --device cpu ...

# 减少batch size
python 4_evaluate_model.py --batch_size 1 ...
```

## 📖 更多信息

- **完整文档**: 查看 `docs/` 目录
- **脚本说明**: 查看 `scripts/README.md`
- **模型支持**: 查看 `models/README.md`
- **配置示例**: 查看 `configs/` 目录

## ❓ 常见问题

### Q: 如何选择mxfp4-ratio?
A: 建议从0.3开始，根据精度-压缩比平衡调整：
- 0.1-0.2: 高压缩，可能损失精度
- 0.3-0.4: 平衡选择（推荐）
- 0.5+: 高精度，压缩率较低

### Q: GPTQ vs RTN?
A: 
- GPTQ: 更高精度，需要校准数据，较慢
- RTN: 更快速，无需校准，精度略低
- 建议：试验用RTN，生产用GPTQ

### Q: 如何评估更多任务?
A: 使用lm-eval支持的任务名：
```bash
python 4_evaluate_model.py --tasks mmlu,gsm8k,truthfulqa,...
```

### Q: 结果保存在哪里?
A: 
- Alpha值: `results/alpha_values.csv`
- 配置: `configs/auto_quant_config.json`
- 模型: `results/quantized_model.pt`
- 评估: `results/eval_results.json`
- 图表: `results/*.png`

---

**开始你的量化之旅！🚀**

