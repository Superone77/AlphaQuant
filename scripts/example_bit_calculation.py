#!/usr/bin/env python3
"""
Bit 数计算功能使用示例
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.calculate_avg_bits import calculate_average_bits, BIT_MAPPING


def example_with_alpha_hill_config():
    """使用 Alpha_Hill 生成的配置示例"""
    print("示例 1: 使用 Alpha_Hill 生成的配置")
    print("=" * 60)
    
    # 模拟 Alpha_Hill 生成的配置
    alpha_hill_config = {
        "default": {
            "wq": "mxfp8",
            "aq": "mxfp8",
            "group_size": 32
        },
        "overrides": [
            {
                "pattern": "model.layers.0.mlp.experts.0.up_proj",
                "wq": "mxfp4",
                "aq": "mxfp4",
                "group_size": 32,
                "alpha_hill": 6.31621404557396,
                "category": "mlp_up"
            },
            {
                "pattern": "model.layers.0.mlp.experts.0.down_proj",
                "wq": "mxfp4",
                "aq": "mxfp4",
                "group_size": 32,
                "alpha_hill": 5.619340177628737,
                "category": "mlp_down"
            },
            {
                "pattern": "model.layers.0.mlp.experts.1.gate_proj",
                "wq": "mxfp8",
                "aq": "mxfp8",
                "group_size": 32,
                "alpha_hill": 3.532005590760851,
                "category": "mlp_gate"
            },
            {
                "pattern": "lm_head",
                "skip": True
            }
        ]
    }
    
    # 模拟模型参数（实际使用时应该从真实模型获取）
    mock_model_params = {
        "model.layers.0.mlp.experts.0.up_proj": 1000000,      # mxfp4
        "model.layers.0.mlp.experts.0.down_proj": 1000000,    # mxfp4
        "model.layers.0.mlp.experts.1.gate_proj": 800000,     # mxfp8
        "model.layers.0.mlp.experts.1.up_proj": 800000,       # default mxfp8
        "model.layers.0.mlp.experts.1.down_proj": 800000,     # default mxfp8
        "lm_head": 500000                                      # skipped (bf16)
    }
    
    print("配置信息:")
    print(f"默认量化格式: {alpha_hill_config['default']['wq']}")
    print("特殊配置:")
    for override in alpha_hill_config['overrides']:
        if override.get('skip'):
            print(f"  {override['pattern']}: 跳过量化 (bf16)")
        else:
            print(f"  {override['pattern']}: {override['wq']} (alpha: {override['alpha_hill']:.2f})")
    
    print("\n模型参数:")
    for layer, params in mock_model_params.items():
        print(f"  {layer}: {params:,} 参数")
    
    # 计算平均 bit 数
    results = calculate_average_bits(alpha_hill_config, mock_model_params)
    
    print("\n计算结果:")
    print(f"总参数量: {results['total_params']:,}")
    print(f"总 Bit 数: {results['total_bits']:,.2f}")
    print(f"平均 Bit 数: {results['average_bits']:.2f}")
    
    # 显示每层的详细信息
    print("\n每层详细信息:")
    print("-" * 80)
    print(f"{'层名':<40} {'参数量':<12} {'Bit数':<10} {'格式':<15}")
    print("-" * 80)
    
    for detail in results["layer_details"]:
        name = detail["name"][:39] + "..." if len(detail["name"]) > 40 else detail["name"]
        print(f"{name:<40} {detail['param_count']:<12,} {detail['avg_bits']:<10.2f} {detail['format']:<15}")


def example_with_mixed_precision():
    """混合精度量化示例"""
    print("\n\n示例 2: 混合精度量化配置")
    print("=" * 60)
    
    # 混合精度配置
    mixed_config = {
        "default": {
            "wq": "mxfp8",
            "aq": "mxfp8",
            "group_size": 128
        },
        "overrides": [
            {
                "pattern": "model.layers.0.*",
                "wq": "mxfp4",
                "group_size": 64
            },
            {
                "pattern": "model.layers.*.q_proj",
                "wq": "mxfp4",
                "group_size": 64
            },
            {
                "pattern": "lm_head",
                "skip": True
            },
            {
                "pattern": "model.embed_tokens",
                "wq": "bf16"
            }
        ]
    }
    
    # 模拟模型参数
    mock_params = {
        "model.embed_tokens": 2000000,           # bf16
        "model.layers.0.self_attn.q_proj": 500000,  # mxfp4
        "model.layers.0.self_attn.k_proj": 500000,  # mxfp4
        "model.layers.0.self_attn.v_proj": 500000,  # mxfp4
        "model.layers.0.self_attn.o_proj": 500000,  # mxfp4
        "model.layers.0.mlp.gate_proj": 800000,     # mxfp4
        "model.layers.0.mlp.up_proj": 800000,       # mxfp4
        "model.layers.0.mlp.down_proj": 800000,     # mxfp4
        "model.layers.1.self_attn.q_proj": 500000,  # mxfp4
        "model.layers.1.self_attn.k_proj": 500000,  # mxfp8 (default)
        "model.layers.1.self_attn.v_proj": 500000,  # mxfp8 (default)
        "model.layers.1.self_attn.o_proj": 500000,  # mxfp8 (default)
        "lm_head": 1000000                          # skipped (bf16)
    }
    
    print("混合精度配置:")
    print(f"默认: {mixed_config['default']['wq']}")
    print("特殊配置:")
    for override in mixed_config['overrides']:
        if override.get('skip'):
            print(f"  {override['pattern']}: 跳过量化")
        else:
            print(f"  {override['pattern']}: {override['wq']}")
    
    # 计算
    results = calculate_average_bits(mixed_config, mock_params)
    
    print(f"\n结果: 平均 {results['average_bits']:.2f} bits/参数")
    
    # 按 bit 数分组显示
    bit_groups = {}
    for detail in results["layer_details"]:
        avg_bits = detail["avg_bits"]
        if avg_bits not in bit_groups:
            bit_groups[avg_bits] = []
        bit_groups[avg_bits].append(detail)
    
    print("\nBit 数分布:")
    for bits in sorted(bit_groups.keys()):
        layers = bit_groups[bits]
        total_params = sum(l["param_count"] for l in layers)
        print(f"  {bits:.2f} bits: {len(layers)} 层, {total_params:,} 参数")


def example_compression_analysis():
    """压缩率分析示例"""
    print("\n\n示例 3: 压缩率分析")
    print("=" * 60)
    
    # 不同配置的对比
    configs = {
        "全精度 (bf16)": {
            "default": {"wq": "bf16", "aq": "bf16"}
        },
        "全量化 (mxfp8)": {
            "default": {"wq": "mxfp8", "aq": "mxfp8"}
        },
        "全量化 (mxfp4)": {
            "default": {"wq": "mxfp4", "aq": "mxfp4"}
        },
        "混合精度": {
            "default": {"wq": "mxfp8", "aq": "mxfp8"},
            "overrides": [
                {"pattern": "model.layers.*.mlp.*", "wq": "mxfp4", "aq": "mxfp4"},
                {"pattern": "lm_head", "skip": True}
            ]
        }
    }
    
    # 模拟模型参数
    mock_params = {
        "model.layers.0.self_attn.q_proj": 500000,
        "model.layers.0.self_attn.k_proj": 500000,
        "model.layers.0.self_attn.v_proj": 500000,
        "model.layers.0.self_attn.o_proj": 500000,
        "model.layers.0.mlp.gate_proj": 800000,
        "model.layers.0.mlp.up_proj": 800000,
        "model.layers.0.mlp.down_proj": 800000,
        "lm_head": 1000000
    }
    
    total_params = sum(mock_params.values())
    print(f"模型总参数量: {total_params:,}")
    print()
    
    print("不同配置的压缩效果对比:")
    print("-" * 80)
    print(f"{'配置名称':<20} {'平均Bit数':<12} {'模型大小(MB)':<15} {'压缩率':<10}")
    print("-" * 80)
    
    for config_name, config in configs.items():
        results = calculate_average_bits(config, mock_params)
        avg_bits = results['average_bits']
        
        # 计算模型大小 (MB)
        model_size_mb = (total_params * avg_bits) / (8 * 1024 * 1024)
        
        # 计算压缩率 (相对于 bf16)
        compression_ratio = 16.0 / avg_bits
        
        print(f"{config_name:<20} {avg_bits:<12.2f} {model_size_mb:<15.2f} {compression_ratio:<10.2f}x")


if __name__ == "__main__":
    print("Bit 数计算功能使用示例")
    print("=" * 80)
    
    try:
        example_with_alpha_hill_config()
        example_with_mixed_precision()
        example_compression_analysis()
        
        print("\n" + "=" * 80)
        print("所有示例演示完成！")
        
    except Exception as e:
        print(f"\n示例演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc() 