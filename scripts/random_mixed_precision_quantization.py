#!/usr/bin/env python3
"""
随机混合精度量化脚本

根据给定的bf16和mxfp4比例，随机选择：
- Experts中的linear层设置为mxfp4
- Attention中的linear层设置为bf16  
- 其余层设置为mxfp8

使用方法:
    python scripts/random_mixed_precision_quantization.py \
        --model "microsoft/DialoGPT-medium" \
        --mxfp4-ratio 0.3 \
        --bf16-ratio 0.2 \
        --output-config configs/random_mixed_quant.json \
        --seed 42
"""

import sys
import os
import json
import argparse
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple, Set
import numpy as np

# 添加项目根目录到 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alphaquant.utils.hf_utils import load_hf_causal_lm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="随机混合精度量化配置生成器")
    parser.add_argument("--model", type=str, required=True, 
                       help="HF model id 或本地模型路径")
    parser.add_argument("--mxfp4-ratio", type=float, default=0.3,
                       help="mxfp4层的比例 (0.0-1.0)")
    parser.add_argument("--bf16-ratio", type=float, default=0.2,
                       help="bf16层的比例 (0.0-1.0)")
    parser.add_argument("--output-config", type=str, required=True,
                       help="输出配置文件的路径")
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子")
    parser.add_argument("--device", type=str, default="cpu",
                       help="设备类型 (cpu/cuda)")
    parser.add_argument("--dtype", type=str, default="fp32",
                       help="模型数据类型 (fp32/fp16/bf16)")
    parser.add_argument("--group-size", type=int, default=32,
                       help="量化组大小")
    parser.add_argument("--verbose", action="store_true",
                       help="详细输出")
    return parser.parse_args()


def identify_layer_categories(model) -> Dict[str, str]:
    """
    识别模型中每层的类别
    
    Returns:
        Dict[str, str]: 层名到类别的映射
        - "expert": Experts中的linear层
        - "attention": Attention中的linear层  
        - "other": 其他linear层
    """
    layer_categories = {}
    
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and hasattr(module.weight, 'shape'):
            # 检查是否为linear层
            if isinstance(module, (torch.nn.Linear, torch.nn.Linear)):
                if "expert" in name.lower() or "experts" in name.lower():
                    layer_categories[name] = "expert"
                elif any(x in name.lower() for x in ["attn", "attention", "q_proj", "k_proj", "v_proj", "o_proj"]):
                    layer_categories[name] = "attention"
                else:
                    layer_categories[name] = "other"
    
    return layer_categories


def create_random_quantization_config(layer_categories: Dict[str, str],
                                    mxfp4_ratio: float,
                                    bf16_ratio: float,
                                    group_size: int = 32,
                                    seed: int = 42) -> Dict[str, Any]:
    """
    创建随机混合精度量化配置
    
    Args:
        layer_categories: 层名到类别的映射
        mxfp4_ratio: mxfp4层的目标比例
        bf16_ratio: bf16层的目标比例
        group_size: 量化组大小
        seed: 随机种子
        
    Returns:
        量化配置字典
    """
    # 设置随机种子
    random.seed(seed)
    np.random.seed(seed)
    
    # 验证比例
    total_ratio = mxfp4_ratio + bf16_ratio
    if total_ratio > 1.0:
        raise ValueError(f"总比例 ({total_ratio:.2f}) 不能超过 1.0. "
                        f"mxfp4_ratio: {mxfp4_ratio:.2f}, bf16_ratio: {bf16_ratio:.2f}")
    
    # 按类别分组层
    expert_layers = [name for name, cat in layer_categories.items() if cat == "expert"]
    attention_layers = [name for name, cat in layer_categories.items() if cat == "attention"]
    other_layers = [name for name, cat in layer_categories.items() if cat == "other"]
    
    print(f"发现层数统计:")
    print(f"  Expert层: {len(expert_layers)}")
    print(f"  Attention层: {len(attention_layers)}")
    print(f"  其他层: {len(other_layers)}")
    print(f"  总计: {len(layer_categories)}")
    
    # 计算每类需要设置的层数
    total_layers = len(layer_categories)
    target_mxfp4_count = int(total_layers * mxfp4_ratio)
    target_bf16_count = int(total_layers * bf16_ratio)
    target_mxfp8_count = total_layers - target_mxfp4_count - target_bf16_count
    
    print(f"\n目标量化分布:")
    print(f"  mxfp4: {target_mxfp4_count} 层 ({mxfp4_ratio:.1%})")
    print(f"  bf16: {target_bf16_count} 层 ({bf16_ratio:.1%})")
    print(f"  mxfp8: {target_mxfp8_count} 层 ({(1-mxfp4_ratio-bf16_ratio):.1%})")
    
    # 创建配置
    config = {
        "default": {
            "wq": "mxfp8",
            "aq": "mxfp8",
            "group_size": group_size
        },
        "overrides": []
    }
    
    # 随机选择Expert层设置为mxfp4
    if expert_layers:
        mxfp4_expert_count = min(target_mxfp4_count, len(expert_layers))
        selected_expert_layers = random.sample(expert_layers, mxfp4_expert_count)
        
        for layer_name in selected_expert_layers:
            config["overrides"].append({
                "pattern": layer_name,
                "wq": "mxfp4",
                "aq": "mxfp4",
                "group_size": group_size,
                "category": "expert",
                "precision": "mxfp4"
            })
        
        print(f"  Expert层设置为mxfp4: {len(selected_expert_layers)} 层")
        target_mxfp4_count -= mxfp4_expert_count
    
    # 随机选择Attention层设置为bf16
    if attention_layers:
        bf16_attention_count = min(target_bf16_count, len(attention_layers))
        selected_attention_layers = random.sample(attention_layers, bf16_attention_count)
        
        for layer_name in selected_attention_layers:
            config["overrides"].append({
                "pattern": layer_name,
                "skip": True,
                "category": "attention",
                "precision": "bf16"
            })
        
        print(f"  Attention层设置为bf16: {len(selected_attention_layers)} 层")
        target_bf16_count -= bf16_attention_count
    
    # 从剩余层中随机选择设置mxfp4
    remaining_layers = []
    for name, cat in layer_categories.items():
        if name not in [override["pattern"] for override in config["overrides"]]:
            remaining_layers.append((name, cat))
    
    if remaining_layers and target_mxfp4_count > 0:
        selected_remaining_mxfp4 = random.sample(remaining_layers, min(target_mxfp4_count, len(remaining_layers)))
        
        for layer_name, category in selected_remaining_mxfp4:
            config["overrides"].append({
                "pattern": layer_name,
                "wq": "mxfp4",
                "aq": "mxfp4",
                "group_size": group_size,
                "category": category,
                "precision": "mxfp4"
            })
        
        print(f"  其他层设置为mxfp4: {len(selected_remaining_mxfp4)} 层")
        target_mxfp4_count -= len(selected_remaining_mxfp4)
        
        # 更新剩余层列表
        remaining_layers = [(name, cat) for name, cat in remaining_layers 
                           if name not in [override["pattern"] for override in config["overrides"]]]
    
    # 从剩余层中随机选择设置bf16
    if remaining_layers and target_bf16_count > 0:
        selected_remaining_bf16 = random.sample(remaining_layers, min(target_bf16_count, len(remaining_layers)))
        
        for layer_name, category in selected_remaining_bf16:
            config["overrides"].append({
                "pattern": layer_name,
                "skip": True,
                "category": category,
                "precision": "bf16"
            })
        
        print(f"  其他层设置为bf16: {len(selected_remaining_bf16)} 层")
        target_bf16_count -= len(selected_remaining_bf16)
        
        # 更新剩余层列表
        remaining_layers = [(name, cat) for name, cat in remaining_layers 
                           if name not in [override["pattern"] for override in config["overrides"]]]
    
    # 剩余层设置为mxfp8 (默认)
    for layer_name, category in remaining_layers:
        config["overrides"].append({
            "pattern": layer_name,
            "wq": "mxfp8",
            "aq": "mxfp8",
            "group_size": group_size,
            "category": category,
            "precision": "mxfp8"
        })
    
    print(f"  剩余层设置为mxfp8: {len(remaining_layers)} 层")
    
    # 添加特殊层处理
    special_patterns = [
        {"pattern": "*lm_head*", "skip": True, "category": "special", "precision": "bf16"},
        {"pattern": "*.mlp.gate*", "skip": True, "category": "special", "precision": "bf16"},
        {"pattern": "*.embed_tokens*", "skip": True, "category": "special", "precision": "bf16"}
    ]
    
    for special in special_patterns:
        config["overrides"].append(special)
    
    # 添加统计信息
    config["summary"] = {
        "total_layers": total_layers,
        "mxfp4_layers": len([o for o in config["overrides"] if o.get("precision") == "mxfp4"]),
        "bf16_layers": len([o for o in config["overrides"] if o.get("precision") == "bf16"]),
        "mxfp8_layers": len([o for o in config["overrides"] if o.get("precision") == "mxfp8"]),
        "mxfp4_ratio": mxfp4_ratio,
        "bf16_ratio": bf16_ratio,
        "mxfp8_ratio": 1.0 - mxfp4_ratio - bf16_ratio,
        "group_size": group_size,
        "seed": seed,
        "randomization_info": {
            "expert_layers_count": len(expert_layers),
            "attention_layers_count": len(attention_layers),
            "other_layers_count": len(other_layers)
        }
    }
    
    return config


def save_config(config: Dict[str, Any], output_path: str) -> None:
    """保存配置到文件"""
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"\n配置已保存到: {output_path}")


def print_config_summary(config: Dict[str, Any]) -> None:
    """打印配置摘要"""
    summary = config["summary"]
    
    print("\n" + "="*60)
    print("随机混合精度量化配置摘要")
    print("="*60)
    print(f"总层数: {summary['total_layers']}")
    print(f"mxfp4层数: {summary['mxfp4_layers']} ({summary['mxfp4_ratio']:.1%})")
    print(f"bf16层数: {summary['bf16_layers']} ({summary['bf16_ratio']:.1%})")
    print(f"mxfp8层数: {summary['mxfp8_layers']} ({summary['mxfp8_ratio']:.1%})")
    print(f"量化组大小: {summary['group_size']}")
    print(f"随机种子: {summary['seed']}")
    
    print(f"\n层类别分布:")
    info = summary["randomization_info"]
    print(f"  Expert层: {info['expert_layers_count']}")
    print(f"  Attention层: {info['attention_layers_count']}")
    print(f"  其他层: {info['other_layers_count']}")


def main():
    args = parse_args()
    
    print("随机混合精度量化配置生成器")
    print("="*50)
    print(f"模型: {args.model}")
    print(f"mxfp4比例: {args.mxfp4_ratio:.1%}")
    print(f"bf16比例: {args.bf16_ratio:.1%}")
    print(f"随机种子: {args.seed}")
    print(f"量化组大小: {args.group_size}")
    
    # 加载模型
    print(f"\n正在加载模型...")
    try:
        model = load_hf_causal_lm(args.model, device=args.device, dtype=args.dtype)
        print(f"模型加载成功: {type(model).__name__}")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return
    
    # 识别层类别
    print(f"\n正在分析模型层结构...")
    layer_categories = identify_layer_categories(model)
    
    if not layer_categories:
        print("未找到可量化的linear层")
        return
    
    # 创建随机量化配置
    print(f"\n正在生成随机量化配置...")
    config = create_random_quantization_config(
        layer_categories=layer_categories,
        mxfp4_ratio=args.mxfp4_ratio,
        bf16_ratio=args.bf16_ratio,
        group_size=args.group_size,
        seed=args.seed
    )
    
    # 保存配置
    save_config(config, args.output_config)
    
    # 打印摘要
    print_config_summary(config)
    
    if args.verbose:
        print(f"\n详细配置:")
        print(json.dumps(config, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main() 