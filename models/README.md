# AlphaQuant Models

这个文件夹包含了独立的模型定义，使得AlphaQuant不受Transformers库版本的影响。

## 📦 包含的模型

### 1. OLMoE (AllenAI)
- **路径**: `models/olmoe/`
- **文件**:
  - `configuration_olmoe.py` - 配置类
  - `modeling_olmoe.py` - 模型实现
  - `__init__.py` - 模块初始化
- **特性**: Top-8 routing, 8个expert
- **用途**: AllenAI的开源MoE模型

### 2. Qwen2-MoE (Alibaba)
- **路径**: `models/qwen_moe_14b_chat/`
- **文件**:
  - `configuration_qwen2_moe.py` - 配置类
  - `modeling_qwen2_moe.py` - 模型实现
  - `__init__.py` - 模块初始化
- **特性**: Top-4 routing + Shared Expert
- **用途**: Qwen系列的MoE模型

### 3. Mixtral (Mistral AI)
- **路径**: `models/mixtral_model/`
- **文件**:
  - `configuration_mixtral.py` - 配置类
  - `modeling_mixtral.py` - 模型实现
  - `__init__.py` - 模块初始化
- **特性**: Top-2 routing, sparse MoE
- **用途**: Mixtral 7B x 8 Expert模型

### 4. DeepSeek-MoE (DeepSeek)
- **路径**: `models/deepseek_moe_16b_chat/`
- **文件**:
  - `configuration_deepseek.py` - 配置类
  - `modeling_deepseek.py` - 模型实现
  - `__init__.py` - 模块初始化
- **特性**: Multi-level routing, 64 experts
- **用途**: DeepSeek的大规模MoE模型

## 🔧 使用方法

### 方法1: 直接导入

```python
from models.olmoe.modeling_olmoe import OlmoeForCausalLM
from models.olmoe.configuration_olmoe import OlmoeConfig

# 使用自定义模型类
config = OlmoeConfig.from_pretrained("allenai/OLMoE-1B-7B-0924")
model = OlmoeForCausalLM.from_pretrained("allenai/OLMoE-1B-7B-0924")
```

### 方法2: 使用AutoModel (推荐)

如果transformers版本已经支持该模型：

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "allenai/OLMoE-1B-7B-0924",
    trust_remote_code=True
)
```

### 方法3: 注册自定义模型到AutoModel

```python
from transformers import AutoConfig, AutoModelForCausalLM
from models.olmoe.configuration_olmoe import OlmoeConfig
from models.olmoe.modeling_olmoe import OlmoeForCausalLM

# 注册
AutoConfig.register("olmoe", OlmoeConfig)
AutoModelForCausalLM.register(OlmoeConfig, OlmoeForCausalLM)

# 然后可以使用AutoModel
model = AutoModelForCausalLM.from_pretrained("allenai/OLMoE-1B-7B-0924")
```

## 🎯 与AlphaQuant GPTQ集成

所有这些模型都已经在AlphaQuant的GPTQ pipeline中得到支持：

```python
from alphaquant.gptq import gptq_quantize_model, detect_moe_architecture
from transformers import AutoModelForCausalLM

# 加载模型
model = AutoModelForCausalLM.from_pretrained("allenai/OLMoE-1B-7B-0924")

# 自动检测架构
arch = detect_moe_architecture(model)  # 返回 'olmoe'

# 量化（自动使用MoE优化）
quantizers = gptq_quantize_model(
    model=model,
    dataloader=calib_data,
    layer_config=config,
    device='cuda'
)
```

## 🔍 模型架构检测

AlphaQuant会自动检测以下MoE架构：

| 模型 | 检测关键词 | Top-K | 特殊处理 |
|------|-----------|-------|----------|
| **OLMoE** | `olmoe`, `olmo` | 8 | Standard routing |
| **Qwen2-MoE** | `qwen`, `qwen2moe` | 4 | Shared expert gate |
| **Mixtral** | `mixtral` | 2 | Standard routing |
| **DeepSeek-MoE** | `deepseek` | Variable | Multi-level routing |

## 📝 Import修复

所有模型文件已经修复了从transformers的相对导入：

```python
# ❌ 旧的 (从transformers库复制)
from ...modeling_utils import PreTrainedModel
from ...configuration_utils import PretrainedConfig

# ✅ 新的 (修复后)
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
```

这样即使将来transformers库更新，这些模型定义仍然可以正常工作。

## 🛠️ 依赖要求

基本要求：
```
torch >= 2.0.0
transformers >= 4.36.0
```

对于某些新特性（如rope_utils），可能需要：
```
transformers >= 4.41.0
```

如果transformers版本较旧，配置文件会自动fallback：

```python
try:
    from transformers.modeling_rope_utils import RopeParameters
except ImportError:
    RopeParameters = None  # Fallback for older versions
```

## 📚 来源

- **OLMoE**: 从最新transformers库复制（4.46+）
- **Qwen2-MoE**: 从transformers 4.37+复制
- **Mixtral**: 从transformers 4.36+复制
- **DeepSeek-MoE**: 从MoEQuant项目复制

## ⚠️ 注意事项

1. **版本兼容性**: 这些模型定义独立于transformers版本，但某些功能可能需要新版本transformers的依赖
2. **命名空间**: 使用`models.`前缀导入以避免与transformers库冲突
3. **更新**: 如果模型定义有重大更新，需要手动同步

## 🔧 维护和更新

如果需要添加新模型或更新现有模型：

1. 从transformers库或官方repo复制最新版本
2. 修复import路径（使用`fix_model_imports.py`脚本）
3. 添加`__init__.py`
4. 更新此README
5. 在`alphaquant/gptq/model_utils.py`中添加检测逻辑
6. 在`alphaquant/gptq/gptq_moe.py`中添加MoE路由处理（如果是MoE模型）

### 使用修复脚本

```bash
# 修复所有模型文件的import
python fix_model_imports.py
```

## 🧪 测试

测试模型是否正常工作：

```python
# 运行测试脚本
python tests/test_model_loading.py
```

或手动测试：

```python
from models.olmoe.modeling_olmoe import OlmoeForCausalLM

try:
    model = OlmoeForCausalLM.from_pretrained(
        "allenai/OLMoE-1B-7B-0924",
        torch_dtype=torch.bfloat16,
        device_map='cpu'
    )
    print("✓ OLMoE模型加载成功")
except Exception as e:
    print(f"✗ 错误: {e}")
```

## 📖 相关文档

- [GPTQ Pipeline README](../GPTQ_PIPELINE_README.md) - GPTQ量化文档
- [MoE Support](../alphaquant/gptq/MOE_SUPPORT.md) - MoE专用优化文档
- [Transformers Documentation](https://huggingface.co/docs/transformers) - 原始transformers文档

