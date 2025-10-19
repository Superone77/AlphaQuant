# 详细样本结果说明

## 概述

每次评估都会生成一个详细的日志文件 (`*_samples.log`)，记录每一题的：
- 问题文本
- 标准答案
- 模型的推理过程
- 模型的最终答案
- 是否正确

## 文件格式

### 详细样本日志 (`*_samples.log`)

日志文件采用易读的文本格式：

```
====================================================================================================
GSM8K Evaluation Samples - 2024-10-19 14:30:22
Total Samples: 256
====================================================================================================

====================================================================================================
Sample #1
====================================================================================================

Question:
Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. 
How many clips did Natalia sell altogether in April and May?

Gold Answer:
Natalia sold 48/2 = <<48/2=24>>24 clips in May.
Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.
#### 72

Model Reasoning:
Let's solve this step-by-step:
1. In April, Natalia sold 48 clips
2. In May, she sold half as many, so 48 / 2 = 24 clips
3. Total clips sold = 48 + 24 = 72

Model Final Answer:
72

Correct: ✓ YES

Full Model Output:
--------------------------------------------------
Let's solve this step-by-step:
1. In April, Natalia sold 48 clips
2. In May, she sold half as many, so 48 / 2 = 24 clips
3. Total clips sold = 48 + 24 = 72
#### 72
--------------------------------------------------

====================================================================================================
Sample #2
====================================================================================================

Question:
Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting...

...

====================================================================================================
SUMMARY
====================================================================================================
Total Samples: 256
Correct: 165
Incorrect: 91
Accuracy: 0.6445 (165/256)
```

### 日志字段说明

每个样本记录包含：
- **Question**: 问题文本
- **Gold Answer**: 标准答案（包含完整推理）
- **Model Reasoning**: 模型的推理过程（`####` 之前）
- **Model Final Answer**: 模型的最终答案（`####` 之后）
- **Correct**: 是否正确（✓ YES / ✗ NO）
- **Full Model Output**: 模型的完整输出

## 使用示例

### 直接查看日志

```bash
# 查看某个实验的日志
cat gsm8k_analysis/results/baseline_samples.log

# 查看特定样本
grep -A 20 "Sample #1" gsm8k_analysis/results/baseline_samples.log

# 查看所有错误的样本
grep -B 2 "✗ NO" gsm8k_analysis/results/baseline_samples.log

# 统计正确和错误数
grep "Correct:" gsm8k_analysis/results/baseline_samples.log | wc -l
grep "✓ YES" gsm8k_analysis/results/baseline_samples.log | wc -l
grep "✗ NO" gsm8k_analysis/results/baseline_samples.log | wc -l

# 查看汇总
tail -n 10 gsm8k_analysis/results/baseline_samples.log
```

### Python 解析日志

```python
import re

def parse_log_file(log_path):
    """解析日志文件并提取样本信息"""
    samples = []
    
    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 按样本分割
    sample_blocks = re.split(r'={100}\nSample #\d+\n={100}', content)
    
    for block in sample_blocks[1:]:  # 跳过开头
        sample = {}
        
        # 提取问题
        question_match = re.search(r'Question:\n(.*?)\n\nGold Answer:', block, re.DOTALL)
        if question_match:
            sample['question'] = question_match.group(1).strip()
        
        # 提取标准答案
        gold_match = re.search(r'Gold Answer:\n(.*?)\n\nModel Reasoning:', block, re.DOTALL)
        if gold_match:
            sample['gold_answer'] = gold_match.group(1).strip()
        
        # 提取模型推理
        reasoning_match = re.search(r'Model Reasoning:\n(.*?)\n\nModel Final Answer:', block, re.DOTALL)
        if reasoning_match:
            sample['model_reasoning'] = reasoning_match.group(1).strip()
        
        # 提取模型答案
        answer_match = re.search(r'Model Final Answer:\n(.*?)\n\nCorrect:', block, re.DOTALL)
        if answer_match:
            sample['model_answer'] = answer_match.group(1).strip()
        
        # 提取正确性
        correct_match = re.search(r'Correct: (✓ YES|✗ NO)', block)
        if correct_match:
            sample['correct'] = correct_match.group(1) == '✓ YES'
        
        samples.append(sample)
    
    return samples

# 使用示例
samples = parse_log_file('gsm8k_analysis/results/baseline_samples.log')
print(f"解析了 {len(samples)} 个样本")

# 分析错误样本
incorrect = [s for s in samples if not s.get('correct', True)]
print(f"错误样本数: {len(incorrect)}")

for sample in incorrect[:3]:
    print(f"\n问题: {sample['question'][:80]}...")
    print(f"模型答案: {sample['model_answer']}")
```

### 对比不同方法

```python
# 解析所有实验的日志
experiments = {
    'baseline': 'gsm8k_analysis/results/baseline_samples.log',
    'mxfp4_rtn': 'gsm8k_analysis/results/mxfp4_rtn_samples.log',
    'gptq_wikitext2': 'gsm8k_analysis/results/gptq_mxfp4_wikitext2_samples.log',
    'gptq_gsm8k': 'gsm8k_analysis/results/gptq_mxfp4_gsm8k_samples.log'
}

results = {}
for name, path in experiments.items():
    results[name] = parse_log_file(path)

# 找出只有 baseline 做对，量化后做错的题
degraded_count = 0
for i in range(len(results['baseline'])):
    baseline_correct = results['baseline'][i].get('correct', False)
    mxfp4_correct = results['mxfp4_rtn'][i].get('correct', False)
    
    if baseline_correct and not mxfp4_correct:
        degraded_count += 1
        print(f"\n题目 {i+1}:")
        print(f"  问题: {results['baseline'][i]['question'][:80]}...")
        print(f"  Baseline答案: {results['baseline'][i]['model_answer']}")
        print(f"  MXFP4答案: {results['mxfp4_rtn'][i]['model_answer']}")

print(f"\n量化导致错误的题目数: {degraded_count}")
```

## 输出文件

运行 pipeline 后，会在 `gsm8k_analysis/results/` 目录下生成：

```
results/
├── baseline_samples.log                     # 详细样本日志 ⭐
├── mxfp4_rtn_samples.log                    # 详细样本日志 ⭐
├── gptq_mxfp4_wikitext2_samples.log         # 详细样本日志 ⭐
├── gptq_mxfp4_gsm8k_samples.log             # 详细样本日志 ⭐
├── comparison.csv                            # 对比表格
└── comparison.txt                            # 详细分析
```

注意：由于 lm_eval 返回结果包含不可序列化的对象，我们不保存 JSON 格式的聚合结果，
而是在控制台输出和日志文件中记录所有信息。

## 常见分析任务

### 1. 统计分析

```bash
# 查看每个实验的准确率
for log in gsm8k_analysis/results/*_samples.log; do
    echo "$(basename $log):"
    tail -n 4 $log | head -n 4
    echo ""
done

# 统计错误样本数
grep "✗ NO" gsm8k_analysis/results/baseline_samples.log | wc -l

# 统计正确样本数
grep "✓ YES" gsm8k_analysis/results/baseline_samples.log | wc -l
```

### 2. 提取特定样本

```bash
# 提取第10个样本
awk '/Sample #10$/,/^===/' gsm8k_analysis/results/baseline_samples.log

# 提取所有错误样本的题号
grep -B 2 "✗ NO" gsm8k_analysis/results/baseline_samples.log | grep "Sample #"
```

### 3. 对比不同方法

```python
# 使用上面的 parse_log_file 函数
baseline_samples = parse_log_file('gsm8k_analysis/results/baseline_samples.log')
mxfp4_samples = parse_log_file('gsm8k_analysis/results/mxfp4_rtn_samples.log')

# 找出量化导致错误的题目
degraded = []
for i in range(len(baseline_samples)):
    if baseline_samples[i].get('correct', False) and not mxfp4_samples[i].get('correct', False):
        degraded.append(i)

print(f"量化导致错误的题目: {len(degraded)} 个")
print(f"题号: {degraded[:10]}")
```

### 4. 分析推理长度

```python
samples = parse_log_file('gsm8k_analysis/results/baseline_samples.log')

# 统计推理长度
reasoning_lengths = [len(s.get('model_reasoning', '')) for s in samples]
print(f"平均推理长度: {sum(reasoning_lengths)/len(reasoning_lengths):.0f} 字符")
print(f"最短: {min(reasoning_lengths)}")
print(f"最长: {max(reasoning_lengths)}")

# 分析正确率与推理长度的关系
correct_lengths = [len(s['model_reasoning']) for s in samples if s.get('correct', False)]
incorrect_lengths = [len(s['model_reasoning']) for s in samples if not s.get('correct', True)]

print(f"\n正确样本平均推理长度: {sum(correct_lengths)/len(correct_lengths):.0f}")
print(f"错误样本平均推理长度: {sum(incorrect_lengths)/len(incorrect_lengths):.0f}")
```

## 注意事项

1. **文件大小**: 每个 `*_samples.log` 文件约 2-5MB（256个样本）
2. **编码**: 使用 UTF-8 编码，支持中文
3. **格式**: 纯文本格式，易于阅读和处理
4. **推理提取**: 依赖 `####` 分隔符，这是 GSM8K 的标准格式

## 为什么用文本日志而不是 JSON？

lm_eval 返回的结果包含 numpy 类型和其他不可序列化的对象，直接保存 JSON 会出错。
文本日志格式更简单、更稳定，且：
- ✅ 易于直接查看（不需要解析工具）
- ✅ 可以用 grep/awk 等命令行工具处理
- ✅ 避免序列化问题
- ✅ 文件更小（纯文本压缩效果好）

## 自定义分析

### 创建分析脚本

```python
# scripts/analyze_gsm8k_logs.py
import re
from pathlib import Path

def parse_log_file(log_path):
    """解析日志文件"""
    # 使用上面的 parse_log_file 函数
    pass

def compare_experiments():
    """对比不同实验"""
    experiments = ['baseline', 'mxfp4_rtn', 'gptq_mxfp4_wikitext2', 'gptq_mxfp4_gsm8k']
    
    for exp in experiments:
        log_path = f'gsm8k_analysis/results/{exp}_samples.log'
        samples = parse_log_file(log_path)
        correct = sum(1 for s in samples if s.get('correct', False))
        print(f"{exp:30s}: {correct}/{len(samples)} = {correct/len(samples):.2%}")

compare_experiments()
```

### 使用 Jupyter Notebook

```python
# notebook/analyze_gsm8k_logs.ipynb
import re
from pathlib import Path

# 解析所有日志
logs = list(Path('gsm8k_analysis/results').glob('*_samples.log'))
print(f"找到 {len(logs)} 个日志文件")

# 对每个文件进行分析
for log_path in logs:
    print(f"\n分析 {log_path.name}...")
    with open(log_path) as f:
        content = f.read()
    
    # 统计
    total = content.count('Sample #')
    correct = content.count('✓ YES')
    incorrect = content.count('✗ NO')
    
    print(f"  总计: {total}")
    print(f"  正确: {correct}")
    print(f"  错误: {incorrect}")
    print(f"  准确率: {correct/total:.2%}")
```

