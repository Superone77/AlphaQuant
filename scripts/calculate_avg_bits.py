#!/usr/bin/env python3
"""
计算量化后模型的平均 bit 数

支持的量化格式：
- mxfp4: 4.25 bits
- mxfp8: 8.25 bits  
- bf16: 16 bits (包括被跳过的层)
"""

import sys
import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np

# 添加项目根目录到 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alphaquant.utils.hf_utils import load_hf_causal_lm


# 量化格式对应的 bit 数
BIT_MAPPING = {
    "mxfp4": 4.25,
    "mxfp8": 8.25,
    "bf16": 16.0,
    "fp16": 16.0,
    "fp32": 32.0
}

# 默认 bit 数（如果配置中没有指定）
DEFAULT_BITS = 16.0


def load_quantization_config(config_path: str) -> Dict[str, Any]:
    """加载量化配置文件"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def get_model_parameters_count(model) -> Dict[str, int]:
    """获取模型中每层的参数量"""
    param_counts = {}
    
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and module.weight is not None:
            # 计算权重参数量
            weight_params = module.weight.numel()
            param_counts[name] = weight_params
            
            # 如果有偏置，也加上
            if hasattr(module, 'bias') and module.bias is not None:
                param_counts[name] += module.bias.numel()
    
    return param_counts


def calculate_layer_bits(layer_config: Dict[str, Any], param_count: int) -> Tuple[float, str]:
    """计算单层的 bit 数"""
    if layer_config.get("skip", False):
        # 被跳过的层按 bf16 计算
        return BIT_MAPPING["bf16"] * param_count, "bf16 (skipped)"
    
    # 获取量化格式
    wq = layer_config.get("wq")
    aq = layer_config.get("aq")
    
    if wq and wq in BIT_MAPPING:
        # 权重量化
        weight_bits = BIT_MAPPING[wq] * param_count
        format_info = f"{wq}"
    else:
        # 默认按 bf16 计算
        weight_bits = BIT_MAPPING["bf16"] * param_count
        format_info = "bf16 (default)"
    
    # 激活量化（如果有的话）
    if aq and aq in BIT_MAPPING:
        # 激活量化通常只影响推理时的计算，不影响模型大小
        # 这里我们只计算权重部分
        pass
    
    return weight_bits, format_info


def calculate_average_bits(config: Dict[str, Any], model_params: Dict[str, int]) -> Dict[str, Any]:
    """计算整个模型的平均 bit 数"""
    
    total_bits = 0.0
    total_params = 0
    layer_details = []
    
    # 处理 overrides 中的层
    if "overrides" in config:
        for override in config["overrides"]:
            pattern = override.get("pattern")
            if not pattern:
                continue
            
            # 查找匹配的模型层
            matched_layers = []
            for layer_name, param_count in model_params.items():
                if pattern_match(layer_name, pattern):
                    matched_layers.append((layer_name, param_count))
            
            if not matched_layers:
                print(f"警告: 配置中的模式 '{pattern}' 没有匹配到任何模型层")
                continue
            
            for layer_name, param_count in matched_layers:
                layer_bits, format_info = calculate_layer_bits(override, param_count)
                total_bits += layer_bits
                total_params += param_count
                
                layer_details.append({
                    "name": layer_name,
                    "pattern": pattern,
                    "param_count": param_count,
                    "bits": layer_bits,
                    "avg_bits": layer_bits / param_count,
                    "format": format_info
                })
    
    # 处理默认配置
    if "default" in config and config["default"]:
        default_config = config["default"]
        # 查找没有被 overrides 覆盖的层
        covered_layers = set(detail["name"] for detail in layer_details)
        
        for layer_name, param_count in model_params.items():
            if layer_name not in covered_layers:
                layer_bits, format_info = calculate_layer_bits(default_config, param_count)
                total_bits += layer_bits
                total_params += param_count
                
                layer_details.append({
                    "name": layer_name,
                    "pattern": "default",
                    "param_count": param_count,
                    "bits": layer_bits,
                    "avg_bits": layer_bits / param_count,
                    "format": format_info
                })
    
    # 计算平均 bit 数
    if total_params > 0:
        average_bits = total_bits / total_params
    else:
        average_bits = 0.0
    
    return {
        "total_bits": total_bits,
        "total_params": total_params,
        "average_bits": average_bits,
        "layer_details": layer_details,
        "summary": {
            "mxfp4_layers": len([d for d in layer_details if "mxfp4" in d["format"]]),
            "mxfp8_layers": len([d for d in layer_details if "mxfp8" in d["format"]]),
            "bf16_layers": len([d for d in layer_details if "bf16" in d["format"]]),
            "skipped_layers": len([d for d in layer_details if "skipped" in d["format"]]),
            "total_layers": len(layer_details)
        }
    }


def pattern_match(layer_name: str, pattern: str) -> bool:
    """简单的模式匹配（支持通配符 *）"""
    if pattern == "*":
        return True
    
    # 将通配符转换为正则表达式
    import re
    regex_pattern = pattern.replace("*", ".*")
    return re.match(regex_pattern, layer_name) is not None


def print_results(results: Dict[str, Any], config_path: str):
    """打印计算结果"""
    print("=" * 80)
    print("量化模型平均 Bit 数计算报告")
    print("=" * 80)
    print(f"配置文件: {config_path}")
    print()
    
    summary = results["summary"]
    print(f"总参数量: {results['total_params']:,}")
    print(f"总 Bit 数: {results['total_bits']:,.2f}")
    print(f"平均 Bit 数: {results['average_bits']:.2f}")
    print()
    
    print("层统计:")
    print(f"  mxfp4 层数: {summary['mxfp4_layers']}")
    print(f"  mxfp8 层数: {summary['mxfp8_layers']}")
    print(f"  bf16 层数: {summary['bf16_layers']}")
    print(f"  跳过层数: {summary['skipped_layers']}")
    print(f"  总层数: {summary['total_layers']}")
    print()
    
    # 按 bit 数分组统计
    bit_groups = {}
    for detail in results["layer_details"]:
        avg_bits = detail["avg_bits"]
        if avg_bits not in bit_groups:
            bit_groups[avg_bits] = []
        bit_groups[avg_bits].append(detail)
    
    print("Bit 数分布:")
    for bits in sorted(bit_groups.keys()):
        layers = bit_groups[bits]
        total_params = sum(l["param_count"] for l in layers)
        print(f"  {bits:.2f} bits: {len(layers)} 层, {total_params:,} 参数")
    print()
    
    # 显示前10层的详细信息
    print("前10层详细信息:")
    print("-" * 80)
    print(f"{'层名':<40} {'参数量':<12} {'Bit数':<10} {'格式':<15}")
    print("-" * 80)
    
    for detail in results["layer_details"][:10]:
        name = detail["name"][:39] + "..." if len(detail["name"]) > 40 else detail["name"]
        print(f"{name:<40} {detail['param_count']:<12,} {detail['avg_bits']:<10.2f} {detail['format']:<15}")
    
    if len(results["layer_details"]) > 10:
        print(f"  ... 还有 {len(results['layer_details']) - 10} 层")


def save_detailed_results(results: Dict[str, Any], output_path: str):
    """保存详细结果到 CSV 文件"""
    import pandas as pd
    
    df = pd.DataFrame(results["layer_details"])
    df.to_csv(output_path, index=False)
    print(f"\n详细结果已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="计算量化后模型的平均 bit 数")
    parser.add_argument("--config", required=True, help="量化配置文件路径")
    parser.add_argument("--model", help="模型路径或ID（可选，用于获取参数量）")
    parser.add_argument("--device", default="cpu", help="设备 (cpu/cuda)")
    parser.add_argument("--dtype", default="fp32", help="数据类型 (fp32/fp16/bf16)")
    parser.add_argument("--output", help="输出CSV文件路径（可选）")
    parser.add_argument("--dry-run", action="store_true", help="仅分析配置，不加载模型")
    
    args = parser.parse_args()
    
    # 加载量化配置
    print(f"加载量化配置: {args.config}")
    config = load_quantization_config(args.config)
    
    if args.dry_run:
        print("Dry-run 模式：仅分析配置文件")
        # 这里可以添加配置文件的静态分析
        return
    
    # 加载模型获取参数量
    if args.model:
        print(f"加载模型: {args.model}")
        try:
            model = load_hf_causal_lm(args.model, device=args.device, dtype=args.dtype)
            model_params = get_model_parameters_count(model)
            print(f"模型加载成功，共 {len(model_params)} 层")
        except Exception as e:
            print(f"模型加载失败: {e}")
            return
    else:
        print("未提供模型路径，无法获取参数量信息")
        return
    
    # 计算平均 bit 数
    print("计算平均 bit 数...")
    results = calculate_average_bits(config, model_params)
    
    # 打印结果
    print_results(results, args.config)
    
    # 保存详细结果
    if args.output:
        save_detailed_results(results, args.output)


if __name__ == "__main__":
    main() 