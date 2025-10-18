# MXFP4 和 MXFP8 Triton Kernels

本项目实现了基于Triton的MXFP4和MXFP8量化kernels，参考了NVFP4的实现模式。

## 文件结构

```
alphaquant/quantizers/kernel/
├── nvfp4_triton.py      # 原有的NVFP4实现
├── mxfp4_triton.py      # 新的MXFP4实现
├── mxfp8_triton.py      # 新的MXFP8实现
├── fp_torch.py           # 更新了导入和接口
└── __init__.py           # 更新了导出
```

## 主要特性

### MXFP4 Kernel
- 支持1-2-1 FP4量化格式
- **只使用per-block scaling策略（无global per-tensor scaling）**
- **Block size为32（与NVFP4的16不同）**
- 支持随机舍入和确定性舍入
- 自动计算scale_per_b

### MXFP8 Kernel
- 支持E4M3和E5M2两种格式
- E4M3: 最大范围448.0
- E5M2: 最大范围57344.0
- **只使用per-block scaling策略（无global per-tensor scaling）**
- **Block size为32（与NVFP4的16不同）**
- 同样支持随机舍入和确定性舍入

## 使用方法

### 直接使用Kernel函数

```python
import torch
from alphaquant.quantizers.kernel import mxfp4_forward, mxfp8_forward

# 准备输入数据
x = torch.randn(2, 32, device="cuda", dtype=torch.float16)

# MXFP4量化
y_mxfp4 = mxfp4_forward(x, stochastic_rounding=True)

# MXFP8 E4M3量化
y_mxfp8_e4m3 = mxfp8_forward(x, format="e4m3", stochastic_rounding=False)

# MXFP8 E5M2量化
y_mxfp8_e5m2 = mxfp8_forward(x, format="e5m2", stochastic_rounding=True)
```

### 通过fp_torch接口使用

```python
from alphaquant.quantizers.kernel.fp_torch import fake_quant_mxfp4, fake_quant_mxfp8

# MXFP4量化
y_mxfp4 = fake_quant_mxfp4(x, stochastic_rounding=True)

# MXFP8量化
y_mxfp8 = fake_quant_mxfp8(x, format="e4m3", stochastic_rounding=False)
```

## 参数说明

### mxfp4_forward
- `x`: 输入CUDA张量 (f16/f32)
- `stochastic_rounding`: 是否使用随机舍入
- **注意**: 不需要scale_per_t参数，只使用per-block scaling

### mxfp8_forward
- `x`: 输入CUDA张量 (f16/f32)
- `format`: "e4m3" 或 "e5m2"
- `stochastic_rounding`: 是否使用随机舍入
- **注意**: 不需要scale_per_t参数，只使用per-block scaling

## 测试

运行测试脚本验证kernels是否正常工作：

```bash
python test_triton_kernels.py
```

## 技术细节

### 量化策略
1. **块级缩放**: 在每个block内计算scale_per_b进行精细调整（无全局缩放）
2. **量化**: 使用1-2-1 FP4或标准FP8量化级别
3. **反缩放**: 恢复原始数值范围

### 性能优化
- 使用Triton JIT编译
- **32元素block大小（MXFP4/MXFP8）vs 16元素（NVFP4）**
- 4个warps并行执行
- 支持CUDA张量的任意形状

### 与NVFP4的差异
- **Block size**: MXFP4/MXFP8使用32，NVFP4使用16
- **Scaling策略**: MXFP4/MXFP8只使用per-block scaling，NVFP4使用per-tensor + per-block scaling
- **计算复杂度**: MXFP4/MXFP8更简单，不需要全局scale计算

## 注意事项

- 所有kernels都需要CUDA环境
- 输入张量必须是连续的CUDA张量
- 支持f16、bf16、f32输入格式
- 输出保持与输入相同的数据类型 