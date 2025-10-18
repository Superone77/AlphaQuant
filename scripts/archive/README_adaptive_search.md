# 自适应量化搜索算法

基于 Alpha_Hill 重要性的自适应量化阈值搜索算法，自动平衡模型性能和压缩率。

## 算法原理

### 核心思想
该算法通过分析模型中每一层的 Alpha_Hill 重要性分数，自动确定最优的量化阈值，在保证模型性能的前提下实现最大程度的压缩。

### 搜索策略
1. **两阶段搜索**：先确定 mxfp4 阈值，再确定 mxfp8 阈值
2. **性能约束**：PPL 增加不超过设定的阈值
3. **压缩目标**：达到目标平均 bit 数
4. **层保护**：自动保护 lm_head 和 mlp.gate 层不被量化

## 算法流程

### 步骤 0: 初始化
- 加载模型
- 计算所有层的 Alpha_Hill 值
- 按重要性分数降序排序

### 步骤 1: 确定 mxfp4 阈值
- 从最大 alpha 值开始
- 逐步降低阈值（步长可调）
- 当 PPL 增加超过阈值时停止
- 记录 mxfp4 阈值

### 步骤 2: 确定 mxfp8 阈值
- 从最大 alpha 值开始
- 逐步降低阈值（步长可调）
- 当 PPL 增加超过阈值或达到最小 alpha 值时停止
- 记录 mxfp8 阈值

### 步骤 3: 输出结果
- 生成量化配置文件
- 生成量化计划文件
- 包含完整的搜索历史

## 使用方法

### 基本用法

```bash
python scripts/adaptive_quantization_search.py \
    --model meta-llama/Llama-3.2-1B \
    --ppl-threshold 0.1 \
    --target-avg-bits 8.0 \
    --output-config config.json \
    --output-plan plan.json
```

### 参数说明

- `--model`: 模型路径或 HuggingFace 模型 ID（必需）
- `--device`: 设备类型，默认 "cpu"
- `--dtype`: 数据类型，默认 "bf16"
- `--ppl-threshold`: PPL 升高的可接受百分比，默认 0.1 (10%)
- `--target-avg-bits`: 目标平均 bit 数，默认 8.0
- `--step-size`: 阈值搜索步长，默认 0.5
- `--batch-size`: 评估时的 batch size，默认 1
- `--output-config`: 输出配置文件路径（必需）
- `--output-plan`: 输出量化计划文件路径（必需）

## 参数调优建议

### PPL 阈值选择
- **保守策略 (0.05)**: 只允许 5% PPL 增加，保证高精度
- **平衡策略 (0.1)**: 允许 10% PPL 增加，平衡精度和压缩
- **激进策略 (0.2)**: 允许 20% PPL 增加，追求高压缩

### 目标 bit 数选择
- **高精度 (10.0-12.0)**: 适合对精度要求高的场景
- **平衡 (8.0-10.0)**: 适合一般应用场景
- **高压缩 (6.0-8.0)**: 适合对模型大小要求严格的场景

### 搜索步长选择
- **精细搜索 (0.3)**: 更精确的阈值，但搜索时间较长
- **标准搜索 (0.5)**: 平衡精度和效率
- **快速搜索 (0.8)**: 快速得到结果，但可能不够精确

## 输出文件

### 量化配置文件
包含完整的量化配置信息，支持通配符模式匹配：

```json
{
  "default": {
    "wq": "bf16",
    "aq": "bf16",
    "group_size": 32
  },
  "overrides": [
    {
      "pattern": "model.layers.0.*",
      "wq": "mxfp4",
      "aq": "mxfp4",
      "group_size": 32,
      "alpha_hill": 8.5,
      "category": "mlp_up"
    },
    {
      "pattern": "lm_head",
      "skip": true
    }
  ],
  "search_summary": {
    "mxfp4_threshold": 8.5,
    "mxfp8_threshold": 5.0,
    "ppl_threshold": 0.1,
    "target_avg_bits": 8.0,
    "step_size": 0.5,
    "search_history": [...]
  }
}
```

### 量化计划文件
可直接用于模型量化的计划文件，格式与配置文件相同。

## 使用示例

### 示例 1: 保守量化策略
```bash
python scripts/adaptive_quantization_search.py \
    --model meta-llama/Llama-3.2-1B \
    --ppl-threshold 0.05 \
    --target-avg-bits 10.0 \
    --step-size 0.3 \
    --output-config conservative_config.json \
    --output-plan conservative_plan.json
```

### 示例 2: 平衡量化策略
```bash
python scripts/adaptive_quantization_search.py \
    --model meta-llama/Llama-3.2-1B \
    --ppl-threshold 0.1 \
    --target-avg-bits 8.0 \
    --step-size 0.5 \
    --output-config balanced_config.json \
    --output-plan balanced_plan.json
```

### 示例 3: 激进量化策略
```bash
python scripts/adaptive_quantization_search.py \
    --model meta-llama/Llama-3.2-1B \
    --ppl-threshold 0.2 \
    --target-avg-bits 6.0 \
    --step-size 0.8 \
    --output-config aggressive_config.json \
    --output-plan aggressive_plan.json
```

## 测试和验证

### 运行测试
```bash
# 运行功能测试
python scripts/test_adaptive_search.py

# 运行使用示例
python scripts/example_adaptive_search.py
```

### 验证结果
搜索完成后，可以通过以下方式验证结果：
1. 检查生成的配置文件格式是否正确
2. 使用 `calculate_avg_bits.py` 验证平均 bit 数
3. 应用量化配置并重新评估 PPL

## 注意事项

### 计算资源要求
- **内存**: 需要足够的内存来加载模型和运行评估
- **时间**: 搜索过程可能需要较长时间，取决于模型大小和搜索步长
- **GPU**: 建议使用 GPU 加速模型评估

### 依赖要求
- transformers: 用于模型加载
- lm-eval: 用于 PPL 评估
- torch: 用于模型操作
- numpy: 用于数值计算

### 最佳实践
1. **从小模型开始**: 先用小模型测试算法效果
2. **调整参数**: 根据具体需求调整 PPL 阈值和目标 bit 数
3. **监控进度**: 搜索过程中监控 PPL 变化和平均 bit 数
4. **验证结果**: 搜索完成后验证量化效果

## 故障排除

### 常见问题

1. **模型加载失败**
   - 检查模型路径是否正确
   - 确保有足够的磁盘空间和内存

2. **PPL 评估失败**
   - 检查 lm-eval 是否正确安装
   - 确保模型格式兼容

3. **搜索时间过长**
   - 减小搜索步长
   - 使用更小的模型进行测试

4. **内存不足**
   - 使用 CPU 而不是 GPU
   - 减小 batch size
   - 使用更小的模型

### 调试模式
使用较小的模型和参数进行测试，确保算法正常工作后再应用到目标模型。

## 扩展功能

### 自定义评估任务
可以修改 `evaluate_ppl` 方法来支持其他评估任务。

### 自定义量化格式
可以在 `bit_mapping` 中添加新的量化格式。

### 自定义搜索策略
可以继承 `AdaptiveQuantizationSearch` 类来实现自定义的搜索策略。 