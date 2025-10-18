# 随机混合精度量化脚本

这个脚本根据给定的bf16和mxfp4比例，随机选择不同层的量化精度，实现智能的混合精度量化策略。

## 功能特点

- **智能层分类**: 自动识别模型中的Expert层、Attention层和其他层
- **随机选择策略**: 根据指定比例随机选择层进行不同精度的量化
- **优先级分配**: 
  - 优先将Expert层设置为mxfp4（高压缩比）
  - 优先将Attention层设置为bf16（保持精度）
  - 其余层设置为mxfp8（平衡压缩和精度）
- **可重现性**: 支持设置随机种子，确保结果可重现
- **灵活配置**: 支持自定义量化组大小和其他参数

## 使用方法

### 基本用法

```bash
python scripts/random_mixed_precision_quantization.py \
    --model "microsoft/DialoGPT-medium" \
    --mxfp4-ratio 0.3 \
    --bf16-ratio 0.2 \
    --output-config configs/random_mixed_quant.json \
    --seed 42
```

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model` | str | 必需 | HF模型ID或本地模型路径 |
| `--mxfp4-ratio` | float | 0.3 | mxfp4层的目标比例 (0.0-1.0) |
| `--bf16-ratio` | float | 0.2 | bf16层的目标比例 (0.0-1.0) |
| `--output-config` | str | 必需 | 输出配置文件的路径 |
| `--seed` | int | 42 | 随机种子，确保结果可重现 |
| `--device` | str | "cpu" | 设备类型 (cpu/cuda) |
| `--dtype` | str | "fp32" | 模型数据类型 (fp32/fp16/bf16) |
| `--group-size` | int | 32 | 量化组大小 |
| `--verbose` | flag | False | 详细输出模式 |

### 量化策略

脚本会按照以下优先级分配量化精度：

1. **Expert层优先mxfp4**: 如果Expert层数量足够，优先将其设置为mxfp4
2. **Attention层优先bf16**: 如果Attention层数量足够，优先将其设置为bf16
3. **剩余层随机分配**: 根据目标比例，从剩余层中随机选择设置mxfp4或bf16
4. **默认mxfp8**: 所有未指定的层默认设置为mxfp8

### 特殊层处理

以下特殊层会被自动设置为bf16（跳过量化）：
- `*lm_head*`: 语言模型输出层
- `*.mlp.gate*`: MLP门控层
- `*.embed_tokens*`: 词嵌入层

## 输出配置格式

生成的配置文件包含以下结构：

```json
{
  "default": {
    "wq": "mxfp8",
    "aq": "mxfp8",
    "group_size": 32
  },
  "overrides": [
    {
      "pattern": "expert1",
      "wq": "mxfp4",
      "aq": "mxfp4",
      "group_size": 32,
      "category": "expert",
      "precision": "mxfp4"
    },
    {
      "pattern": "q_proj",
      "skip": true,
      "category": "attention",
      "precision": "bf16"
    }
  ],
  "summary": {
    "total_layers": 10,
    "mxfp4_layers": 3,
    "bf16_layers": 2,
    "mxfp8_layers": 5,
    "mxfp4_ratio": 0.3,
    "bf16_ratio": 0.2,
    "mxfp8_ratio": 0.5,
    "group_size": 32,
    "seed": 42
  }
}
```

## 使用示例

### 示例1: 高压缩比配置

```bash
# 30% mxfp4, 10% bf16, 60% mxfp8
python scripts/random_mixed_precision_quantization.py \
    --model "microsoft/DialoGPT-medium" \
    --mxfp4-ratio 0.3 \
    --bf16-ratio 0.1 \
    --output-config configs/high_compression.json \
    --seed 123
```

### 示例2: 平衡配置

```bash
# 25% mxfp4, 25% bf16, 50% mxfp8
python scripts/random_mixed_precision_quantization.py \
    --model "microsoft/DialoGPT-medium" \
    --mxfp4-ratio 0.25 \
    --bf16-ratio 0.25 \
    --output-config configs/balanced.json \
    --seed 456
```

### 示例3: 高精度配置

```bash
# 10% mxfp4, 40% bf16, 50% mxfp8
python scripts/random_mixed_precision_quantization.py \
    --model "microsoft/DialoGPT-medium" \
    --mxfp4-ratio 0.1 \
    --bf16-ratio 0.4 \
    --output-config configs/high_precision.json \
    --seed 789
```

## 测试

运行测试脚本来验证功能：

```bash
python scripts/test_random_mixed_precision.py
```

测试包括：
- 层分类功能测试
- 随机配置生成测试
- 配置保存和加载测试
- 边界情况测试

## 注意事项

1. **比例限制**: `mxfp4_ratio + bf16_ratio` 不能超过1.0
2. **层数量**: 实际设置的层数可能与目标比例略有差异，取决于模型结构
3. **随机性**: 相同种子会产生相同结果，不同种子会产生不同结果
4. **模型兼容性**: 脚本支持大多数基于Transformer的模型架构

## 与其他脚本的集成

生成的配置文件可以用于：

1. **量化模型**: 使用 `scripts/quantize_model.py` 进行实际量化
2. **性能评估**: 使用 `scripts/eval_with_lm_eval.py` 评估量化后性能
3. **位宽计算**: 使用 `scripts/calculate_avg_bits.py` 计算平均位宽

## 故障排除

### 常见问题

1. **模型加载失败**: 检查模型路径和网络连接
2. **层分类为空**: 确保模型包含Linear层
3. **比例验证失败**: 检查mxfp4_ratio + bf16_ratio <= 1.0

### 调试模式

使用 `--verbose` 参数获取详细输出：

```bash
python scripts/random_mixed_precision_quantization.py \
    --model "your_model" \
    --mxfp4-ratio 0.3 \
    --bf16-ratio 0.2 \
    --output-config configs/debug.json \
    --verbose
``` 