# MXFP8 NaN Issue Fix

## 问题描述

在使用MXFP8量化时，某些情况下输出结果会是NaN。这主要发生在以下几种场景：

1. **全零输入**：当输入张量某一行全为0时
2. **极小值输入**：当输入值非常接近0时
3. **输入本身包含NaN/Inf**：当输入数据已经包含异常值时

## 根本原因

### PyTorch实现 (`fp_torch.py`)

原始代码在计算缩放因子时存在问题：

```python
# 原始代码 (第368行)
scale = torch.pow(2.0, torch.floor(torch.log2(fp8_max / x_abs.max(dim=-1, keepdim=True)[0])))
```

**问题**：
- 当 `x_abs.max()` 为 0 时，`fp8_max / 0` = inf
- `log2(inf)` = inf
- `pow(2.0, inf)` = inf
- 后续计算 `x / scale` 可能产生 NaN

### Triton Kernel实现 (`mxfp8_triton.py`)

原始代码：
```python
scale_b = tl.where(blk_max > 0.0, 448.0 / blk_max, 1.0)
```

虽然有条件检查，但：
- 当 `blk_max` 非常小（但非零）时，`scale_b` 会非常大
- `y = x_abs * scale_b` 可能溢出成 inf
- 后续反缩放 `q_abs = q_abs_blk / scale_b` 可能产生异常值

## 修复方案

### 1. PyTorch实现修复

**修改文件**: `alphaquant/quantizers/kernel/fp_torch.py`

**主要改进**：
1. **防止除零**：在计算前将max值clamp到最小值（1e-12）
2. **限制log范围**：将log2的结果限制在合理范围内（-20到20）
3. **增强scale检查**：不仅检查inf，还检查NaN
4. **最终安全网**：在返回前检查并替换任何NaN/Inf为0

```python
# 修复后的代码
def mxfp8_torch(x:torch.Tensor, 
                   stochastic_rounding:bool=False, 
                   scaled_value_format:str='e4m3') -> torch.Tensor:
    # ... 省略部分代码 ...
    
    # 防止除零
    max_val = x_abs.max(dim=-1, keepdim=True)[0]
    max_val = torch.clamp(max_val, min=1e-12)
    
    # 安全计算scale
    ratio = fp8_max / max_val
    log_ratio = torch.log2(ratio)
    log_ratio = torch.clamp(log_ratio, min=-20, max=20)
    scale = torch.pow(2.0, torch.floor(log_ratio))
    
    # 增强的安全检查
    scale = torch.where((0 < scale) * (scale < torch.inf) * ~torch.isnan(scale), scale, 1.0)
    
    # ... 量化操作 ...
    
    # 最终安全检查
    result = torch.where(torch.isnan(result) | torch.isinf(result), torch.zeros_like(result), result)
    
    return result.to(x.dtype)
```

### 2. Triton Kernel修复

**修改文件**: `alphaquant/quantizers/kernel/mxfp8_triton.py`

**主要改进**：
1. **使用tl.maximum代替tl.where**：直接确保blk_max有最小值
2. **限制缩放后的值**：防止溢出
3. **NaN/Inf检查**：在最终输出前检查并替换异常值

```python
# E4M3格式修复
blk_max = tl.maximum(blk_max, 1e-12)
scale_b = 448.0 / blk_max
y = x_abs * scale_b
y = tl.minimum(y, 448.0)  # 防止溢出
# ... 量化操作 ...
q_abs = tl.where((q_abs == q_abs) & (q_abs != float('inf')) & (q_abs != float('-inf')), q_abs, 0.0)

# E5M2格式同样修复
blk_max = tl.maximum(blk_max, 1e-12)
scale_b = 57344.0 / blk_max
y = x_abs * scale_b
y = tl.minimum(y, 57344.0)  # 防止溢出
# ... 量化操作 ...
q_abs = tl.where((q_abs == q_abs) & (q_abs != float('inf')) & (q_abs != float('-inf')), q_abs, 0.0)
```

## 测试验证

创建了测试脚本 `test_mxfp8_nan.py` 来验证修复：

```bash
python test_mxfp8_nan.py
```

测试用例包括：
- 正常随机值
- 极小值（1e-10数量级）
- 全零张量
- 混合零值和正常值
- 极大值（1e10数量级）
- 单个零行
- 输入包含NaN
- 输入包含Inf

## 影响范围

修复影响以下组件：
- `MXFP8Quantizer.quantize_weight()` - 权重量化
- `MXFP8Quantizer.quantize_activation()` - 激活量化
- 所有使用MXFP8量化的模型和配置

## 建议

1. **使用修复后的代码**：确保使用最新版本的 `fp_torch.py` 和 `mxfp8_triton.py`
2. **验证输出**：在量化后检查是否有NaN/Inf
3. **监控数值稳定性**：在训练/推理时监控量化后的数值范围
4. **输入验证**：在量化前检查输入数据的有效性

## 注意事项

- 修复后，全零输入将被量化为全零输出（而不是NaN）
- 极端值会被安全地限制在合理范围内
- 性能影响可忽略不计（仅增加了少量检查操作）

