# Group Quantization 功能实现总结

## 修改概述

已成功为 INT 和 FP 类型（除 MX 类型外）的量化器添加 group quantization 支持。

## 修改的文件

### 1. 量化器实现文件

#### `/alphaquant/quantizers/int_quantizers.py`
**修改内容**:
- 在所有 INT Config 类中添加 `use_group_quant` 和 `group_size` 参数
  - `INT2Config`, `INT3Config`, `INT4Config`, `INT6Config`, `INT8Config`
  - 默认 `use_group_quant=False`, `group_size=128`
- 在 `BaseIntQuantizer` 中添加:
  - `use_group_quant` 和 `group_size` 属性
  - `_quantize_weight_group()` 方法: 实现分组量化逻辑
  - 修改 `quantize_weight()` 方法: 根据 `use_group_quant` 选择量化方式

**核心功能**:
```python
def _quantize_weight_group(self, w: torch.Tensor):
    """对权重进行分组量化"""
    # 1. 展平张量
    # 2. 按 group_size 分组（带填充）
    # 3. 为每组计算独立的 scale 和 zero_point
    # 4. 量化每组
    # 5. 重组回原始形状
```

#### `/alphaquant/quantizers/fp4.py`
**修改内容**:
- 在 `FP4Config` 中添加 `use_group_quant` 参数（默认 False）
- 在 `FP4Quantizer` 中添加:
  - `use_group_quant` 和 `group_size` 属性
  - `_quantize_weight_group()` 方法
  - 修改 `quantize_weight()` 方法

#### `/alphaquant/quantizers/fp8.py`
**修改内容**:
- 在 `FP8Config` 中添加 `use_group_quant` 参数（默认 False）
- 在 `FP8Quantizer` 中添加:
  - `use_group_quant` 和 `group_size` 属性
  - `_quantize_weight_group()` 方法
  - 修改 `quantize_weight()` 方法

### 2. 示例和文档文件

#### `/example_group_quantization.py` (新建)
完整的演示脚本，展示:
- INT4, INT8 量化器的 group quantization
- FP4, FP8 量化器的 group quantization
- 有/无 group quantization 的对比
- 量化误差分析

#### `/run_group_quantization_example.sh` (新建)
便捷的 shell 脚本，用于运行示例

#### `/test_group_quantization.py` (新建)
快速测试脚本，验证所有量化器的 group quantization 功能

#### `/GROUP_QUANTIZATION_README.md` (新建)
详细的使用文档，包括:
- 功能概述
- 参数说明
- 使用示例
- 工作原理
- 性能考虑
- 常见问题

#### `/GROUP_QUANTIZATION_SUMMARY.md` (本文件)
修改总结和快速参考

## 关键参数

### `use_group_quant`
- **类型**: `bool`
- **默认值**: `False`
- **说明**: 控制是否启用 group quantization

### `group_size`
- **类型**: `int`
- **默认值**: `128`
- **说明**: 每组的元素数量

## 快速使用示例

### INT4 with Group Quantization
```python
from alphaquant.quantizers.int_quantizers import INT4Config, INT4Quantizer
import torch

# 创建配置（启用 group quantization）
config = INT4Config(
    use_group_quant=True,  # 启用
    group_size=128,        # 组大小
    symmetric=True
)

# 创建量化器
quantizer = INT4Quantizer(config)

# 量化权重
weight = torch.randn(1024, 1024)
quantized_weight = quantizer.quantize_weight(weight)
```

### FP8 with Group Quantization
```python
from alphaquant.quantizers.fp8 import FP8Config, FP8Quantizer

config = FP8Config(
    use_group_quant=True,
    group_size=128,
    format="e4m3"
)

quantizer = FP8Quantizer(config)
quantized_weight = quantizer.quantize_weight(weight)
```

## 运行示例

### 运行完整示例
```bash
# 方式 1: Python 脚本
python example_group_quantization.py

# 方式 2: Shell 脚本
bash run_group_quantization_example.sh
```

### 运行快速测试
```bash
python test_group_quantization.py
```

## 支持的量化器

### ✅ 支持 Group Quantization
- INT2, INT3, INT4, INT6, INT8
- FP4, FP8

### ❌ 不支持
- MXFP4, MXFP8 (MX 类型量化器有自己的分组机制)

## 设计特点

1. **向后兼容**: 默认 `use_group_quant=False`，不影响现有代码
2. **灵活配置**: `group_size` 可自定义，适应不同场景
3. **统一接口**: 所有量化器使用相同的参数和方法
4. **自动填充**: 自动处理不能整除 group_size 的情况

## 实现细节

### 分组量化流程
1. **展平**: `w.flatten()` 将权重展平为一维
2. **填充**: 如果需要，填充到 group_size 的整数倍
3. **重组**: `reshape(num_groups, group_size)`
4. **计算**: 为每组计算 scale（和可选的 zero_point）
5. **量化**: 使用各组参数分别量化
6. **还原**: 去除填充，恢复原始形状

### 对称量化
```python
# 为每组计算
scale = max(abs(group)) / qmax
quantized = quantize_kernel(group / scale)
dequantized = quantized * scale
```

### 非对称量化（仅 INT 量化器）
```python
# 为每组计算
scale = (max - min) / (qmax - qmin)
zero_point = qmin - min / scale
quantized = quantize_kernel(group / scale + zero_point)
dequantized = (quantized - zero_point) * scale
```

## 性能特征

### 优点
- ✅ 提高量化精度（特别是权重范围差异大时）
- ✅ 更灵活的量化粒度
- ✅ 保持张量形状不变
- ✅ 易于集成到现有流程

### 权衡
- ⚠️ 增加计算开销（需为每组计算参数）
- ⚠️ 增加内存开销（存储多组参数）
- ⚠️ 推理时需要额外的反量化逻辑

### 推荐的 group_size

| 场景 | 推荐值 | 理由 |
|-----|--------|------|
| 精度优先 | 64-128 | 更精细的分组 |
| 平衡 | 128 | **默认推荐** |
| 速度优先 | 256-512 | 减少分组数量 |
| 小权重 | < 64 | 避免组数过少 |

## 文件清单

### 核心修改
- ✅ `alphaquant/quantizers/int_quantizers.py`
- ✅ `alphaquant/quantizers/fp4.py`
- ✅ `alphaquant/quantizers/fp8.py`

### 示例和文档
- ✅ `example_group_quantization.py`
- ✅ `run_group_quantization_example.sh`
- ✅ `test_group_quantization.py`
- ✅ `GROUP_QUANTIZATION_README.md`
- ✅ `GROUP_QUANTIZATION_SUMMARY.md`

## 兼容性

- ✅ 与现有代码完全兼容（默认禁用）
- ✅ 不改变 API 接口
- ✅ 可以与其他量化配置共存
- ✅ 支持 symmetric 和 asymmetric 模式（INT 量化器）

## 下一步

建议的使用流程:
1. 阅读 `GROUP_QUANTIZATION_README.md` 了解详细信息
2. 运行 `example_group_quantization.py` 查看效果
3. 运行 `test_group_quantization.py` 验证功能
4. 在自己的模型中尝试使用，从 `group_size=128` 开始
5. 根据精度和速度需求调整 `group_size`

## 注意事项

1. **首次使用**: 建议从默认值开始（`group_size=128`）
2. **性能测试**: 在生产环境前测试实际影响
3. **激活量化**: 当前实现主要针对权重，激活量化仍使用原有方式
4. **MX 类型**: 不要在 MX 量化器上使用此功能
5. **梯度**: fake quantization 支持梯度传播，可用于 QAT

## 测试建议

```bash
# 1. 快速功能测试
python test_group_quantization.py

# 2. 查看详细对比
python example_group_quantization.py

# 3. 在实际模型上测试
# 修改你的量化配置，添加:
# use_group_quant=True, group_size=128
```

---

**实现时间**: 2024
**支持的量化器**: INT2/3/4/6/8, FP4, FP8
**默认行为**: 禁用（保持向后兼容）

