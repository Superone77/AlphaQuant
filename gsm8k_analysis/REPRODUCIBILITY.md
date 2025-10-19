# GSM8K Analysis - 可重现性说明

## 🎯 确保实验可对比的关键设计

### 1. 确定性样本选择

**重要**: 所有实验评估的是**完全相同**的 256 个样本！

#### lm_eval 的 `limit` 参数

```python
results = evaluator.simple_evaluate(
    model=lm,
    tasks=["gsm8k"],
    limit=256  # 确定性地选择前 256 个样本（不是随机选择）
)
```

**工作原理**:
- `limit=N` 参数会选择 GSM8K 测试集的**前 N 个样本**
- 这是**确定性**的，不是随机选择
- 所有实验都使用相同的 limit 值，因此评估相同的样本
- 样本顺序由数据集本身决定（HuggingFace datasets 的加载顺序是固定的）

#### 验证方法

可以通过查看日志文件验证：

```bash
# 提取所有实验的第1题问题
grep -A 1 "Sample #1" gsm8k_analysis/results/baseline_samples.log | grep "Question:"
grep -A 1 "Sample #1" gsm8k_analysis/results/mxfp4_rtn_samples.log | grep "Question:"
grep -A 1 "Sample #1" gsm8k_analysis/results/gptq_mxfp4_wikitext2_samples.log | grep "Question:"
grep -A 1 "Sample #1" gsm8k_analysis/results/gptq_mxfp4_gsm8k_samples.log | grep "Question:"

# 它们应该完全一样！
```

### 2. 固定随机种子

虽然样本选择是确定性的，但我们仍然设置随机种子以确保：

#### 为什么需要随机种子？

1. **校准数据采样**: GPTQ 的校准数据（WikiText2/GSM8K）需要随机采样
2. **模型推理**: 某些模型配置可能使用 sampling（虽然 GSM8K 通常用 greedy）
3. **PyTorch/CUDA 随机性**: 某些操作可能有随机性
4. **量化过程**: GPTQ 算法中可能有随机元素

#### 随机种子设置

所有脚本都调用 `set_seed(42)`:

```python
def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)              # Python random
    np.random.seed(seed)           # NumPy random
    torch.manual_seed(seed)        # PyTorch CPU random
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # PyTorch GPU random
    # Make CUDA operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

**设置了**:
- ✅ Python `random` 模块
- ✅ NumPy 随机数生成器
- ✅ PyTorch CPU 随机数生成器
- ✅ PyTorch GPU 随机数生成器
- ✅ cuDNN 确定性模式

### 3. 校准数据的可重现性

#### WikiText2 校准

在 `alphaquant/gptq/data_utils.py`:

```python
def get_wikitext2(nsamples=128, seed=0, seqlen=2048, tokenizer=None):
    random.seed(seed)  # 固定随机种子
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        trainloader.append(inp)
```

- 使用固定的 `seed=0`
- 每次采样相同的文本片段

#### GSM8K 校准

在 `gsm8k_analysis/data_utils.py`:

```python
def get_gsm8k_calibration(nsamples=128, seed=0, ...):
    random.seed(seed)  # 固定随机种子
    indices = list(range(len(dataset)))
    random.shuffle(indices)  # 固定顺序的随机打乱
    
    for idx in indices[:nsamples]:
        # 选择样本
```

- 使用固定的 `seed=0`
- 每次选择相同的训练样本

### 4. 确保一致性的检查清单

运行实验前确认：

- [x] 所有脚本使用相同的 `--seed` 参数（默认 42）
- [x] 所有评估使用相同的 `--num_samples` 参数（默认 256）
- [x] 所有 GPTQ 校准使用相同的 `--num_calibration_samples`（默认 128）
- [x] 使用 `limit` 参数而不是随机采样
- [x] 校准数据加载函数使用固定种子

### 5. 完整性验证

#### 验证评估样本一致性

```python
import re

def extract_questions_from_log(log_path):
    """提取日志中所有问题"""
    with open(log_path, 'r') as f:
        content = f.read()
    
    questions = re.findall(r'Question:\n(.*?)\n\nGold Answer:', content, re.DOTALL)
    return [q.strip() for q in questions]

# 检查所有实验的问题是否一致
baseline_q = extract_questions_from_log('gsm8k_analysis/results/baseline_samples.log')
mxfp4_q = extract_questions_from_log('gsm8k_analysis/results/mxfp4_rtn_samples.log')
gptq_wiki_q = extract_questions_from_log('gsm8k_analysis/results/gptq_mxfp4_wikitext2_samples.log')
gptq_gsm8k_q = extract_questions_from_log('gsm8k_analysis/results/gptq_mxfp4_gsm8k_samples.log')

# 验证
assert baseline_q == mxfp4_q, "MXFP4 使用了不同的样本！"
assert baseline_q == gptq_wiki_q, "GPTQ WikiText2 使用了不同的样本！"
assert baseline_q == gptq_gsm8k_q, "GPTQ GSM8K 使用了不同的样本！"

print(f"✓ 验证通过！所有实验评估了相同的 {len(baseline_q)} 个样本")
```

#### 验证校准数据一致性

对于使用相同校准数据的实验（如两次运行 GPTQ + WikiText2），校准数据也应该相同：

```bash
# 如果多次运行 step 3，校准数据应该完全一致
# 这是因为 CalibrationDataLoader 使用固定的 seed=0
```

## 📊 实验设计保证

### 对比是有意义的

```
实验 A: Baseline         → 评估样本 [0, 1, 2, ..., 255]
实验 B: MXFP4 (RTN)      → 评估样本 [0, 1, 2, ..., 255]  ← 相同！
实验 C: GPTQ + WikiText2 → 评估样本 [0, 1, 2, ..., 255]  ← 相同！
实验 D: GPTQ + GSM8K     → 评估样本 [0, 1, 2, ..., 255]  ← 相同！

校准数据也是固定的：
实验 C 第1次运行 → 校准样本 [随机但固定的 128 个 WikiText2 片段]
实验 C 第2次运行 → 校准样本 [完全相同的 128 个片段]  ← 可重现！
```

### 为什么用"前 N 个"而不是"随机 N 个"？

**优点**:
1. ✅ **确定性**: 不需要保存样本索引，自然可重现
2. ✅ **简单**: 不需要额外的采样代码
3. ✅ **标准**: lm_eval 的标准做法
4. ✅ **可验证**: 可以直接对比日志文件

**缺点**:
1. ⚠️ 可能不够有代表性（如果数据集有顺序偏差）

**GSM8K 情况**:
- GSM8K 测试集是打乱的，没有明显顺序偏差
- 前 256 个样本具有代表性
- 大多数研究也是用前 N 个样本

## 🔬 进阶：完全控制样本选择

如果需要**更精细的控制**（如随机采样但固定种子），可以修改：

```python
# 在评估脚本中添加
import random
from datasets import load_dataset

def get_fixed_sample_indices(num_samples=256, seed=42):
    """获取固定的随机样本索引"""
    dataset = load_dataset('gsm8k', 'main', split='test')
    random.seed(seed)
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    return indices[:num_samples]

# 使用固定索引
sample_indices = get_fixed_sample_indices(256, seed=42)

# 然后在 lm_eval 中使用这些索引...
# (需要修改 lm_eval 的调用方式)
```

但这需要修改 lm_eval 的使用方式，当前的 `limit` 方法更简单且足够。

## ✅ 总结

**当前实现的可重现性保证**:

| 要素 | 方法 | 状态 |
|------|------|------|
| 评估样本选择 | `limit` 参数（前 N 个） | ✅ 确定性 |
| 随机种子 | `set_seed(42)` | ✅ 已设置 |
| 校准数据采样 | 固定 seed=0 | ✅ 确定性 |
| CUDA 确定性 | `cudnn.deterministic=True` | ✅ 已设置 |
| 模型推理 | Greedy decoding（默认） | ✅ 确定性 |

**结论**: 
- ✅ 所有实验评估**完全相同**的 256 个 GSM8K 样本
- ✅ 实验结果是**可重现**的
- ✅ 对比是**有意义**的

## 🧪 验证脚本

运行以下脚本验证实验一致性：

```python
# gsm8k_analysis/verify_consistency.py
import re

def verify_same_samples():
    """验证所有实验使用了相同的样本"""
    logs = [
        'gsm8k_analysis/results/baseline_samples.log',
        'gsm8k_analysis/results/mxfp4_rtn_samples.log',
        'gsm8k_analysis/results/gptq_mxfp4_wikitext2_samples.log',
        'gsm8k_analysis/results/gptq_mxfp4_gsm8k_samples.log'
    ]
    
    all_questions = []
    
    for log_path in logs:
        with open(log_path, 'r') as f:
            content = f.read()
        questions = re.findall(r'Question:\n(.*?)\n\nGold Answer:', content, re.DOTALL)
        all_questions.append([q.strip() for q in questions])
        print(f"✓ {log_path}: {len(questions)} 个样本")
    
    # 验证所有实验的问题列表完全一致
    for i in range(1, len(all_questions)):
        if all_questions[0] != all_questions[i]:
            print(f"✗ 错误！{logs[i]} 使用了不同的样本")
            return False
    
    print(f"\n✓ 验证通过！所有实验评估了相同的 {len(all_questions[0])} 个样本")
    return True

if __name__ == '__main__':
    verify_same_samples()
```

使用方法：

```bash
python gsm8k_analysis/verify_consistency.py
```

