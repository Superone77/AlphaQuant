#!/usr/bin/env python3
"""
测试 bit 数计算功能
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.calculate_avg_bits import (
    calculate_layer_bits, 
    calculate_average_bits,
    BIT_MAPPING
)


def test_layer_bits_calculation():
    """测试单层 bit 数计算"""
    print("测试单层 bit 数计算")
    print("=" * 50)
    
    # 测试数据
    test_cases = [
        {
            "config": {"wq": "mxfp4", "aq": "mxfp4"},
            "param_count": 1000000,
            "expected_bits": 4.25,
            "description": "mxfp4 量化"
        },
        {
            "config": {"wq": "mxfp8", "aq": "mxfp8"},
            "param_count": 1000000,
            "expected_bits": 8.25,
            "description": "mxfp8 量化"
        },
        {
            "config": {"skip": True},
            "param_count": 1000000,
            "expected_bits": 16.0,
            "description": "跳过层（按 bf16 计算）"
        },
        {
            "config": {"wq": "bf16"},
            "param_count": 1000000,
            "expected_bits": 16.0,
            "description": "bf16 格式"
        },
        {
            "config": {},
            "param_count": 1000000,
            "expected_bits": 16.0,
            "description": "默认配置（按 bf16 计算）"
        }
    ]
    
    for i, test_case in enumerate(test_cases):
        print(f"\n测试 {i+1}: {test_case['description']}")
        print(f"配置: {test_case['config']}")
        print(f"参数量: {test_case['param_count']:,}")
        
        bits, format_info = calculate_layer_bits(test_case['config'], test_case['param_count'])
        avg_bits = bits / test_case['param_count']
        
        print(f"计算结果: {bits:,.2f} bits, 平均: {avg_bits:.2f} bits/param")
        print(f"格式信息: {format_info}")
        
        # 验证结果
        expected_total = test_case['expected_bits'] * test_case['param_count']
        if abs(bits - expected_total) < 0.01:
            print("✓ 测试通过")
        else:
            print(f"✗ 测试失败: 期望 {expected_total:,.2f}, 实际 {bits:,.2f}")


def test_average_bits_calculation():
    """测试平均 bit 数计算"""
    print("\n\n测试平均 bit 数计算")
    print("=" * 50)
    
    # 模拟量化配置
    mock_config = {
        "default": {"wq": "mxfp8", "aq": "mxfp8"},
        "overrides": [
            {
                "pattern": "layer_1",
                "wq": "mxfp4",
                "aq": "mxfp4"
            },
            {
                "pattern": "layer_2",
                "skip": True
            },
            {
                "pattern": "layer_3",
                "wq": "bf16",
                "aq": "bf16"
            }
        ]
    }
    
    # 模拟模型参数
    mock_model_params = {
        "layer_1": 500000,    # mxfp4: 4.25 bits
        "layer_2": 300000,    # skipped: 16.0 bits
        "layer_3": 400000,    # bf16: 16.0 bits
        "layer_4": 600000,    # default mxfp8: 8.25 bits
        "layer_5": 700000     # default mxfp8: 8.25 bits
    }
    
    print("模拟配置:")
    print(f"默认: {mock_config['default']}")
    print("覆盖:")
    for override in mock_config['overrides']:
        print(f"  {override['pattern']}: {override}")
    
    print("\n模拟模型参数:")
    for layer, params in mock_model_params.items():
        print(f"  {layer}: {params:,} 参数")
    
    # 计算平均 bit 数
    results = calculate_average_bits(mock_config, mock_model_params)
    
    print("\n计算结果:")
    print(f"总参数量: {results['total_params']:,}")
    print(f"总 Bit 数: {results['total_bits']:,.2f}")
    print(f"平均 Bit 数: {results['average_bits']:.2f}")
    
    # 手动验证计算
    expected_total_bits = (
        500000 * 4.25 +    # layer_1: mxfp4
        300000 * 16.0 +    # layer_2: skipped (bf16)
        400000 * 16.0 +    # layer_3: bf16
        600000 * 8.25 +    # layer_4: default mxfp8
        700000 * 8.25      # layer_5: default mxfp8
    )
    
    expected_avg_bits = expected_total_bits / sum(mock_model_params.values())
    
    print(f"\n手动验证:")
    print(f"期望总 Bit 数: {expected_total_bits:,.2f}")
    print(f"期望平均 Bit 数: {expected_avg_bits:.2f}")
    
    if abs(results['total_bits'] - expected_total_bits) < 0.01:
        print("✓ 总 Bit 数计算正确")
    else:
        print("✗ 总 Bit 数计算错误")
    
    if abs(results['average_bits'] - expected_avg_bits) < 0.01:
        print("✓ 平均 Bit 数计算正确")
    else:
        print("✗ 平均 Bit 数计算错误")


def test_bit_mapping():
    """测试 bit 映射"""
    print("\n\n测试 Bit 映射")
    print("=" * 50)
    
    print("支持的量化格式及其 bit 数:")
    for format_name, bits in BIT_MAPPING.items():
        print(f"  {format_name}: {bits} bits")
    
    # 验证一些关键值
    assert BIT_MAPPING["mxfp4"] == 4.25, "mxfp4 应该是 4.25 bits"
    assert BIT_MAPPING["mxfp8"] == 8.25, "mxfp8 应该是 8.25 bits"
    assert BIT_MAPPING["bf16"] == 16.0, "bf16 应该是 16.0 bits"
    print("✓ Bit 映射验证通过")


if __name__ == "__main__":
    print("Bit 数计算功能测试")
    print("=" * 60)
    
    try:
        test_bit_mapping()
        test_layer_bits_calculation()
        test_average_bits_calculation()
        
        print("\n" + "=" * 60)
        print("所有测试完成！")
        
    except Exception as e:
        print(f"\n测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc() 