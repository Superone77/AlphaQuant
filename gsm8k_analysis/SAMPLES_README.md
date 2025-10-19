# 详细样本结果说明

## 概述

每次评估都会生成两个 JSON 文件：
1. **聚合结果** (`*_results.json`): 包含整体准确率等统计指标
2. **详细样本** (`*_samples.json`): 包含每一题的详细信息

## 文件格式

### 详细样本文件 (`*_samples.json`)

每个样本包含以下字段：

```json
{
  "timestamp": "2024-10-19T...",
  "task": "gsm8k",
  "statistics": {
    "total_samples": 256,
    "correct": 165,
    "incorrect": 91,
    "accuracy": 0.6445
  },
  "samples": [
    {
      "index": 0,
      "doc_id": 0,
      "question": "Natalia sold clips to 48 of her friends...",
      "gold_answer": "#### 60",
      "model_reasoning": "Let's solve this step-by-step:\n1. In April, she sold 48 clips\n2. In May, she sold 48 / 2 = 24 clips\n3. Total = 48 + 24 = 72 clips",
      "model_final_answer": "72",
      "model_full_output": "Let's solve this step-by-step:\n1. In April, she sold 48 clips\n2. In May, she sold 48 / 2 = 24 clips\n3. Total = 48 + 24 = 72 clips\n#### 72",
      "correct": false,
      "metrics": {
        "exact_match,strict-match": false
      }
    }
  ]
}
```

### 字段说明

- **index**: 样本序号（0开始）
- **doc_id**: 数据集中的原始ID
- **question**: 问题文本
- **gold_answer**: 标准答案（通常格式为 `#### 数字`）
- **model_reasoning**: 模型的推理过程（`####` 之前的部分）
- **model_final_answer**: 模型的最终答案（`####` 之后的部分）
- **model_full_output**: 模型的完整输出
- **correct**: 是否正确（true/false）
- **metrics**: 其他评估指标

## 使用示例

### Python 读取和分析

```python
import json
import pandas as pd

# 读取详细样本
with open('gsm8k_analysis/results/baseline_samples.json') as f:
    data = json.load(f)

# 转换为 DataFrame
df = pd.DataFrame(data['samples'])

# 查看错误样本
incorrect = df[df['correct'] == False]
print(f"错误样本数: {len(incorrect)}")
print(incorrect[['question', 'gold_answer', 'model_final_answer']])

# 分析推理长度
df['reasoning_length'] = df['model_reasoning'].str.len()
print(f"平均推理长度: {df['reasoning_length'].mean():.0f} 字符")

# 找出特定类型的错误
for idx, row in incorrect.head(5).iterrows():
    print(f"\n问题 {row['index']}:")
    print(f"题目: {row['question'][:100]}...")
    print(f"标准答案: {row['gold_answer']}")
    print(f"模型答案: {row['model_final_answer']}")
    print(f"推理过程: {row['model_reasoning'][:200]}...")
```

### 对比不同方法

```python
import json

# 读取多个实验的结果
experiments = {
    'baseline': 'gsm8k_analysis/results/baseline_samples.json',
    'mxfp4_rtn': 'gsm8k_analysis/results/mxfp4_rtn_samples.json',
    'gptq_wikitext2': 'gsm8k_analysis/results/gptq_mxfp4_wikitext2_samples.json',
    'gptq_gsm8k': 'gsm8k_analysis/results/gptq_mxfp4_gsm8k_samples.json'
}

results = {}
for name, path in experiments.items():
    with open(path) as f:
        results[name] = json.load(f)

# 找出只有 baseline 做对，量化后做错的题
baseline_samples = {s['index']: s for s in results['baseline']['samples']}
mxfp4_samples = {s['index']: s for s in results['mxfp4_rtn']['samples']}

degraded = []
for idx in baseline_samples:
    if baseline_samples[idx]['correct'] and not mxfp4_samples[idx]['correct']:
        degraded.append({
            'index': idx,
            'question': baseline_samples[idx]['question'],
            'baseline_answer': baseline_samples[idx]['model_final_answer'],
            'mxfp4_answer': mxfp4_samples[idx]['model_final_answer']
        })

print(f"\n量化导致错误的题目数: {len(degraded)}")
for item in degraded[:3]:
    print(f"\n题目 {item['index']}: {item['question'][:80]}...")
    print(f"  Baseline答案: {item['baseline_answer']}")
    print(f"  MXFP4答案: {item['mxfp4_answer']}")
```

### 分析推理模式

```python
import re
from collections import Counter

# 读取结果
with open('gsm8k_analysis/results/baseline_samples.json') as f:
    data = json.load(f)

# 分析推理步骤数
def count_steps(reasoning):
    # 统计包含数字和运算符的行
    steps = re.findall(r'.*[+\-*/=].*', reasoning)
    return len(steps)

step_counts = [count_steps(s['model_reasoning']) for s in data['samples']]
print(f"平均推理步骤数: {sum(step_counts)/len(step_counts):.1f}")

# 分析正确率与推理长度的关系
import pandas as pd
df = pd.DataFrame(data['samples'])
df['step_count'] = [count_steps(s['model_reasoning']) for s in data['samples']]

print("\n推理步骤数与正确率:")
for steps in range(1, 6):
    subset = df[df['step_count'] == steps]
    if len(subset) > 0:
        acc = subset['correct'].mean()
        print(f"  {steps} 步: {acc:.2%} ({len(subset)} 题)")
```

## 输出文件

运行 pipeline 后，会在 `gsm8k_analysis/results/` 目录下生成：

```
results/
├── baseline.json                        # 聚合结果
├── baseline_samples.json               # 详细样本 ⭐
├── mxfp4_rtn.json
├── mxfp4_rtn_samples.json              # 详细样本 ⭐
├── gptq_mxfp4_wikitext2.json
├── gptq_mxfp4_wikitext2_samples.json   # 详细样本 ⭐
├── gptq_mxfp4_gsm8k.json
├── gptq_mxfp4_gsm8k_samples.json       # 详细样本 ⭐
├── comparison.csv
└── comparison.txt
```

## 常见分析任务

### 1. 找出最难的题

```python
# 统计哪些题所有方法都做错
all_wrong = []
for idx in range(256):
    wrong_count = sum([
        not results[exp]['samples'][idx]['correct'] 
        for exp in ['baseline', 'mxfp4_rtn', 'gptq_wikitext2', 'gptq_gsm8k']
    ])
    if wrong_count == 4:
        all_wrong.append(results['baseline']['samples'][idx])

print(f"所有方法都做错的题目: {len(all_wrong)}")
```

### 2. 分析量化的影响模式

```python
# 看看量化主要影响哪类题目
error_types = {
    'baseline_only': [],  # 只有baseline对
    'all_correct': [],    # 所有方法都对
    'quant_improved': []  # 量化后反而对了
}

for idx in range(256):
    b = results['baseline']['samples'][idx]['correct']
    m = results['mxfp4_rtn']['samples'][idx]['correct']
    
    if b and not m:
        error_types['baseline_only'].append(idx)
    elif b and m:
        error_types['all_correct'].append(idx)
    elif not b and m:
        error_types['quant_improved'].append(idx)

for k, v in error_types.items():
    print(f"{k}: {len(v)} 题")
```

### 3. 导出为 CSV 用于人工分析

```python
import pandas as pd

# 读取所有实验
dfs = []
for name, path in experiments.items():
    with open(path) as f:
        data = json.load(f)
    df = pd.DataFrame(data['samples'])
    df['experiment'] = name
    dfs.append(df[['index', 'question', 'gold_answer', 'model_final_answer', 'correct', 'experiment']])

# 合并并导出
combined = pd.concat(dfs)
combined.to_csv('gsm8k_analysis/results/all_samples_comparison.csv', index=False)
print("已导出到 all_samples_comparison.csv")
```

## 注意事项

1. **文件大小**: 每个 `*_samples.json` 文件约 1-2MB（256个样本）
2. **编码**: 使用 UTF-8 编码，支持中文
3. **格式**: 标准 JSON，可以用任何 JSON 工具打开
4. **推理提取**: 依赖 `####` 分隔符，这是 GSM8K 的标准格式

## 自定义分析

如果需要自定义分析，可以：

1. 修改 `gsm8k_analysis/eval_utils.py` 中的提取逻辑
2. 创建自己的分析脚本
3. 使用 Jupyter Notebook 进行交互式分析

示例 Notebook:

```python
# notebook/analyze_gsm8k_samples.ipynb
import json
import pandas as pd
import matplotlib.pyplot as plt

# 加载数据
with open('../gsm8k_analysis/results/baseline_samples.json') as f:
    baseline = json.load(f)

# 可视化
df = pd.DataFrame(baseline['samples'])
df['reasoning_length'] = df['model_reasoning'].str.len()

plt.figure(figsize=(10, 6))
plt.scatter(df['reasoning_length'], df['correct'], alpha=0.5)
plt.xlabel('Reasoning Length')
plt.ylabel('Correct')
plt.title('Reasoning Length vs Correctness')
plt.savefig('reasoning_analysis.png')
```

