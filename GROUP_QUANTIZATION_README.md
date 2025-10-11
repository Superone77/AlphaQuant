# Group Quantization 使用指南

## 概述

Group Quantization（分组量化）是一种改进的量化技术，它将张量内的元素分组后分别计算量化参数。这种方法对于具有不同数值范围的权重特别有效，能够显著提高量化精度。

本项目已为以下量化器添加了 Group Quantization 支持：
- **INT 量化器**: INT2, INT3, INT4, INT6, INT8
- **FP 量化器**: FP4, FP8
- **注意**: MX 类型量化器不支持此功能

## 主要参数

### `use_group_quant`
- **类型**: `bool`
- **默认值**: `False`
- **说明**: 控制是否启用 group quantization
  - `False`: 使用标准的 per-tensor quantization
  - `True`: 使用 group quantization

### `group_size`
- **类型**: `int`
- **默认值**: `128`
- **说明**: 每组的元素数量
  - 较小的 group_size (如 64, 128) 提供更精细的量化，但会增加开销
  - 较大的 group_size (如 256, 512) 减少开销，但可能降低精度

## 使用方法

### 1. INT 量化器示例

#### INT4 量化
```python
from alphaquant.quantizers.int_quantizers import INT4Config, INT4Quantizer
import torch

# 创建测试权重
weight = torch.randn(1024, 1024)

# 启用 group quantization
config = INT4Config(
    use_group_quant=True,  # 启用 group quantization
    group_size=128,        # 每组 128 个元素
    symmetric=True         # 使用对称量化
)

quantizer = INT4Quantizer(config)
quantized_weight = quantizer.quantize_weight(weight)
```

#### INT8 量化
```python
from alphaquant.quantizers.int_quantizers import INT8Config, INT8Quantizer

# 使用更大的 group size
config = INT8Config(
    use_group_quant=True,
    group_size=256,        # 每组 256 个元素
    symmetric=True
)

quantizer = INT8Quantizer(config)
quantized_weight = quantizer.quantize_weight(weight)
```

### 2. FP 量化器示例

#### FP4 量化
```python
from alphaquant.quantizers.fp4 import FP4Config, FP4Quantizer

config = FP4Config(
    use_group_quant=True,
    group_size=128,
    format="e2m1"          # FP4 格式
)

quantizer = FP4Quantizer(config)
quantized_weight = quantizer.quantize_weight(weight)
```

#### FP8 量化
```python
from alphaquant.quantizers.fp8 import FP8Config, FP8Quantizer

config = FP8Config(
    use_group_quant=True,
    group_size=128,
    format="e4m3"          # FP8 E4M3 格式
)

quantizer = FP8Quantizer(config)
quantized_weight = quantizer.quantize_weight(weight)
```

### 3. 在配置文件中使用

如果你使用 JSON 配置文件，可以这样设置：

```json
{
  "quantizers": {
    "weight_quantizer": {
      "type": "int4",
      "use_group_quant": true,
      "group_size": 128,
      "symmetric": true
    }
  }
}
```

## 运行示例

### 方法 1: 使用 Python 脚本
```bash
python example_group_quantization.py
```

### 方法 2: 使用 Shell 脚本
```bash
bash run_group_quantization_example.sh
# 或者
./run_group_quantization_example.sh
```

## 工作原理

### 标准量化 (Per-tensor)
标准量化为整个张量计算单一的量化参数（scale 和 zero point）：

```
Tensor [N elements] → Single scale → Quantized tensor
```

### Group Quantization
Group quantization 将张量分成多个组，每组分别计算量化参数：

```
Tensor [N elements] → [Group 1, Group 2, ..., Group K]
                      ↓        ↓              ↓
                   Scale 1, Scale 2, ..., Scale K
                      ↓        ↓              ↓
           Quantized [Group 1, Group 2, ..., Group K]
```

### 算法步骤

1. **展平**: 将输入张量展平为一维
2. **分组**: 按 `group_size` 分成多个组（最后一组可能需要填充）
3. **计算参数**: 为每个组分别计算 scale 和 zero_point
4. **量化**: 使用各组的参数分别量化
5. **重组**: 将结果重组回原始形状

## 性能考虑

### 精度 vs. 开销权衡

| Group Size | 精度 | 计算开销 | 内存开销 | 推荐场景 |
|-----------|------|---------|---------|---------|
| 64        | 最高 | 最高    | 最高    | 权重差异大的小模型 |
| 128       | 高   | 中等    | 中等    | **推荐默认值** |
| 256       | 中   | 低      | 低      | 大模型，追求速度 |
| 512+      | 低   | 很低    | 很低    | 接近 per-tensor |

### 何时使用 Group Quantization

✅ **推荐使用的场景**:
- 权重在不同区域有明显不同的数值范围
- 对量化精度要求较高
- 有足够的计算和内存资源

❌ **不推荐使用的场景**:
- 权重分布相对均匀
- 极度追求推理速度
- 内存资源受限
- 使用 MX 类型量化器

## 技术细节

### 对称量化 (Symmetric)
```python
# 为每组计算
scale = max(abs(group)) / qmax
quantized = round(group / scale)
dequantized = quantized * scale
```

### 非对称量化 (Asymmetric)
```python
# 为每组计算
scale = (max(group) - min(group)) / (qmax - qmin)
zero_point = qmin - min(group) / scale
quantized = round(group / scale + zero_point)
dequantized = (quantized - zero_point) * scale
```

## 所有支持的量化器

### INT 量化器
- `INT2Config` / `INT2Quantizer`
- `INT3Config` / `INT3Quantizer`
- `INT4Config` / `INT4Quantizer`
- `INT6Config` / `INT6Quantizer`
- `INT8Config` / `INT8Quantizer`

### FP 量化器
- `FP4Config` / `FP4Quantizer`
  - 格式: `e2m1`
- `FP8Config` / `FP8Quantizer`
  - 格式: `e4m3`, `e5m2`

### 不支持的量化器
- `MXFP4` (MX 类型)
- `MXFP8` (MX 类型)

## 常见问题

### Q: 为什么 MX 类型不支持 group quantization?
A: MX 类型量化器已经有自己的分组机制（block-wise quantization），与这里实现的 group quantization 概念不同。

### Q: group_size 应该设置多大？
A: 默认值 128 在大多数情况下是一个好的选择。如果你的权重有明显的局部模式，可以尝试更小的值（64）；如果追求速度，可以使用更大的值（256）。

### Q: group quantization 会影响推理速度吗？
A: 是的，group quantization 会增加一些计算开销，因为需要为每个组维护单独的 scale。但这种开销通常是可以接受的，特别是在追求高精度的场景下。

### Q: 可以在激活量化中使用 group quantization 吗？
A: 目前 group quantization 主要针对权重量化实现。激活量化仍然使用原有的 per-tensor 或 per-channel 方式。

## 参考资料

- 相关文件:
  - `alphaquant/quantizers/int_quantizers.py`
  - `alphaquant/quantizers/fp4.py`
  - `alphaquant/quantizers/fp8.py`
- 示例脚本:
  - `example_group_quantization.py`
  - `run_group_quantization_example.sh`

## 更新日志

- **2024**: 添加 group quantization 支持
  - INT2, INT3, INT4, INT6, INT8 量化器
  - FP4, FP8 量化器
  - 默认 group_size = 128
  - 默认 use_group_quant = False（保持向后兼容）

