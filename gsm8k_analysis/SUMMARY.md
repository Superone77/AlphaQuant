# GSM8K Analysis Pipeline - 项目总结

## 📋 项目概述

这个 pipeline 用于系统地分析 OLMoE 模型在 GSM8K 数据集上的量化精度损失问题。

### 核心功能

1. **Baseline 评估**: 在 256 个 GSM8K 样本上评估原始模型性能
2. **MXFP4 量化**: 将所有 expert 层量化到 MXFP4 并评估
3. **GPTQ + WikiText2**: 使用 WikiText2 作为校准数据进行 GPTQ 量化
4. **GPTQ + GSM8K**: 使用 GSM8K 训练集作为校准数据进行 GPTQ 量化
5. **结果对比分析**: 自动生成对比表格和分析报告

## 📁 文件结构

```
gsm8k_analysis/
├── README.md              # 完整文档
├── QUICKSTART.md          # 快速开始指南
├── SUMMARY.md             # 本文件 - 项目总结
├── __init__.py            # 模块初始化
│
├── run_pipeline.py        # 主 pipeline 脚本（Python）
├── run_pipeline.sh        # 主 pipeline 脚本（Bash）
├── example_usage.py       # 使用示例
│
├── data_utils.py          # GSM8K 数据加载工具
├── eval_baseline.py       # Baseline 评估脚本
├── quantize_mxfp4.py      # MXFP4 量化脚本
├── quantize_gptq.py       # GPTQ 量化脚本
├── analyze_results.py     # 结果分析脚本
│
├── configs/               # 量化配置（自动生成）
│   ├── mxfp4_plan_*.json
│   ├── gptq_wikitext2_plan_*.json
│   └── gptq_gsm8k_plan_*.json
│
└── results/               # 评估结果（自动生成）
    ├── baseline_*.json
    ├── mxfp4_*.json
    ├── gptq_wikitext2_*.json
    ├── gptq_gsm8k_*.json
    ├── pipeline_summary_*.json
    ├── comparison_*.csv
    └── comparison_*.txt
```

## 🚀 快速使用

### 一键运行

```bash
cd /Users/superone77/Code/AlphaQuant
bash gsm8k_analysis/run_pipeline.sh
```

### 自定义参数

```bash
bash gsm8k_analysis/run_pipeline.sh \
    allenai/OLMoE-1B-7B-0924 \  # 模型
    256 \                        # GSM8K 样本数
    128 \                        # 校准样本数
    cuda \                       # 设备
    bfloat16 \                   # 数据类型
    1                            # batch size
```

## 🔧 技术细节

### 1. 数据加载 (`data_utils.py`)

**功能**:
- 从 HuggingFace 加载 GSM8K 数据集
- 支持作为 GPTQ 校准数据
- 灵活的采样和分割选项

**关键函数**:
- `get_gsm8k_calibration()`: 获取校准数据
- `get_gsm8k_test_samples()`: 获取测试样本索引
- `GSM8KCalibrationDataLoader`: 数据加载器类

### 2. Baseline 评估 (`eval_baseline.py`)

**功能**:
- 评估原始（未量化）模型
- 作为后续对比的基准

**输出**:
- JSON 格式的评估结果
- 包含准确率和标准误差

### 3. MXFP4 量化 (`quantize_mxfp4.py`)

**量化方案**:
- 格式: MXFP4 with e8m0
- 目标: 所有 expert 层（w1, w2, w3）
- Group size: 128
- 无激活量化

**特点**:
- 快速、无需校准
- 自动检测 expert 层
- 保存量化计划

### 4. GPTQ 量化 (`quantize_gptq.py`)

**量化方案**:
- 算法: GPTQ
- Bits: 4-bit (可配置)
- Group size: 128
- 支持多种校准数据集

**配置参数**:
- `percdamp`: 0.01
- `blocksize`: 128
- `actorder`: 可选
- `static_groups`: False

**支持的校准数据**:
- WikiText2 (通用文本)
- GSM8K (任务特定)
- C4 (大规模文本)

### 5. 结果分析 (`analyze_results.py`)

**功能**:
- 解析所有实验结果
- 生成对比表格
- 计算精度损失
- 识别最佳方法

**输出格式**:
- CSV: 机器可读的对比表
- TXT: 人类可读的详细分析

## 📊 预期结果示例

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

Key Findings:
✓ GPTQ with task-specific calibration (GSM8K) performs best
✓ Generic calibration (WikiText2) shows 45% more accuracy loss
✓ MXFP4 without calibration has highest accuracy drop
```

## 🎯 设计原则

### 1. 解耦性
- 每个脚本独立运行
- 清晰的接口和参数
- 可单独测试和调试

### 2. 可复用性
- 利用 AlphaQuant 现有基础设施
- 遵循项目代码风格
- 不重复造轮子

### 3. 可扩展性
- 易于添加新的量化方法
- 支持新的校准数据集
- 灵活的参数配置

### 4. 可重现性
- 固定随机种子
- 保存所有配置
- 详细的日志输出

## 🔍 代码组织

### 依赖关系

```
run_pipeline.py
    ├── eval_baseline.py
    │   └── lm_eval.evaluator
    │
    ├── quantize_mxfp4.py
    │   ├── alphaquant.utils.replacement
    │   └── lm_eval.evaluator
    │
    ├── quantize_gptq.py
    │   ├── alphaquant.gptq.quantize
    │   ├── data_utils.py (local)
    │   └── lm_eval.evaluator
    │
    └── analyze_results.py
        └── pandas

data_utils.py
    ├── datasets.load_dataset
    └── transformers.PreTrainedTokenizer
```

### 外部依赖

**来自 AlphaQuant**:
- `alphaquant.gptq.*`: GPTQ 量化
- `alphaquant.quantizers.*`: 各种量化器
- `alphaquant.utils.*`: 工具函数

**来自第三方**:
- `lm_eval`: 模型评估
- `transformers`: 模型加载
- `datasets`: 数据集加载
- `pandas`: 数据分析

## ⚙️ 运行时配置

### 默认参数

```python
MODEL = "allenai/OLMoE-1B-7B-0924"
NUM_SAMPLES = 256
NUM_CALIBRATION_SAMPLES = 128
BATCH_SIZE = 1
DEVICE = "cuda"
DTYPE = "bfloat16"
SEQLEN = 2048
```

### 资源需求

**内存**:
- GPU: ~40-50GB (A100/H100)
- CPU RAM: ~32GB

**存储**:
- 模型缓存: ~15GB
- 结果文件: ~50MB

**时间**:
- Baseline: ~10-15分钟
- MXFP4: ~15-20分钟
- GPTQ (每个): ~30-45分钟
- 总计: ~2-3小时

## 🛠️ 维护和扩展

### 添加新的量化方法

1. 创建新脚本 `quantize_<method>.py`
2. 参考 `quantize_mxfp4.py` 的结构
3. 在 `run_pipeline.py` 中添加步骤
4. 更新 `analyze_results.py`（如需要）

### 添加新的校准数据集

1. 在 `data_utils.py` 添加加载函数
2. 更新 `quantize_gptq.py` 的选项
3. 测试数据加载

### 自定义分析

编辑 `analyze_results.py` 添加:
- 新的指标
- 可视化图表
- 统计检验

## 📝 使用注意事项

### 常见问题

1. **OOM (内存溢出)**
   - 减少样本数
   - 减少 batch size
   - 使用 CPU offload

2. **速度慢**
   - 增加 batch size
   - 减少样本数
   - 使用更快的 GPU

3. **结果不一致**
   - 检查随机种子
   - 确认数据加载顺序
   - 验证量化参数

### 最佳实践

1. **首次运行**: 使用少量样本测试（如 10-20）
2. **调试**: 使用 `--skip_*` 参数跳过已完成的步骤
3. **对比**: 保持其他参数不变，只改变一个变量
4. **记录**: 保存所有配置和结果

## 🎓 学习资源

### 相关论文

- GPTQ: [paper link]
- OLMoE: [paper link]
- GSM8K: [paper link]

### 代码参考

- AlphaQuant GPTQ 实现: `alphaquant/gptq/`
- 量化器实现: `alphaquant/quantizers/`
- 原始 pipeline: `run_pipeline.sh`

## 📞 支持

遇到问题？

1. 查看 `QUICKSTART.md` 快速开始指南
2. 查看 `README.md` 详细文档
3. 运行 `example_usage.py` 查看示例
4. 使用 `--help` 查看各脚本参数

## ✅ 项目完成度

- [x] GSM8K 数据加载工具
- [x] Baseline 评估脚本
- [x] MXFP4 量化脚本
- [x] GPTQ 量化脚本（支持多种校准数据）
- [x] 主 pipeline 脚本（Python + Bash）
- [x] 结果分析脚本
- [x] 完整文档（README + QUICKSTART + SUMMARY）
- [x] 使用示例
- [x] 代码组织和解耦
- [x] 无 linter 错误

## 🚀 未来改进

可能的扩展方向：

1. **更多量化方法**: INT8, FP8, AWQ, SmoothQuant
2. **混合精度**: 基于敏感度的逐层量化
3. **更多数据集**: MATH, ARC, TriviaQA
4. **可视化**: 准确率曲线、热图、分布图
5. **自动调参**: 网格搜索、贝叶斯优化
6. **模型对比**: 不同 OLMoE 版本
7. **详细分析**: 每层影响、expert 利用率

---

**创建时间**: 2024-10-19  
**版本**: 1.0  
**作者**: AlphaQuant Team

