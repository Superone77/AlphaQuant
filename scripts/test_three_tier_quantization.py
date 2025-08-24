#!/usr/bin/env python3
"""
测试三层量化策略：mxfp4（高alpha）、mxfp8（中等alpha）、bf16（低alpha，保持原精度）
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.alpha_hill_quantization import create_quantization_config


def test_three_tier_quantization():
    """测试三层量化策略"""
    print("测试三层量化策略")
    print("=" * 50)
    
    # 模拟 Alpha_Hill 结果
    mock_alpha_results = {
        f"layer_{i}": {
            'alpha': 1.0 - i * 0.1,  # 从 1.0 递减到 0.1
            'category': 'linear',
            'out_features': 512,
            'in_features': 512,
            'numel': 262144,
            'dtype': 'float32',
            'device': 'cpu'
        }
        for i in range(10)
    }
    
    print("模拟的 Alpha_Hill 值:")
    for name, result in mock_alpha_results.items():
        print(f"  {name}: alpha={result['alpha']:.3f}")
    
    # 测试不同的比例配置
    test_configs = [
        {"mxfp4_ratio": 0.3, "bf16_ratio": 0.2},  # 30% mxfp4, 20% bf16, 50% mxfp8
        {"mxfp4_ratio": 0.5, "bf16_ratio": 0.3},  # 50% mxfp4, 30% bf16, 20% mxfp8
        {"mxfp4_ratio": 0.2, "bf16_ratio": 0.1},  # 20% mxfp4, 10% bf16, 70% mxfp8
        {"mxfp4_ratio": 0.0, "bf16_ratio": 0.5},  # 0% mxfp4, 50% bf16, 50% mxfp8
    ]
    
    for i, config in enumerate(test_configs):
        print(f"\n测试配置 {i+1}: mxfp4_ratio={config['mxfp4_ratio']:.1f}, bf16_ratio={config['bf16_ratio']:.1f}")
        print("-" * 60)
        
        try:
            result = create_quantization_config(
                mock_alpha_results,
                mxfp4_ratio=config['mxfp4_ratio'],
                bf16_ratio=config['bf16_ratio']
            )
            
            summary = result['summary']
            print(f"总层数: {summary['total_layers']}")
            print(f"mxfp4 层数: {summary['mxfp4_layers']} ({summary['mxfp4_ratio']:.1%})")
            print(f"mxfp8 层数: {summary['mxfp8_layers']} ({summary['mxfp8_ratio']:.1%})")
            print(f"bf16 层数: {summary['bf16_layers']} ({summary['bf16_ratio']:.1%})")
            print(f"mxfp4 阈值: {summary['mxfp4_threshold']:.4f}")
            print(f"bf16 阈值: {summary['bf16_threshold']:.4f}")
            
            # 显示每层的分配
            print("\n层分配详情:")
            for override in result['overrides']:
                if override.get('skip'):
                    precision = "bf16 (skip)"
                elif override.get('wq') == 'mxfp4':
                    precision = "mxfp4"
                elif override.get('wq') == 'mxfp8':
                    precision = "mxfp8"
                else:
                    precision = "unknown"
                
                print(f"  {override['pattern']}: {precision}, alpha={override['alpha_hill']:.4f}")
                
        except Exception as e:
            print(f"错误: {e}")


def test_validation():
    """测试参数验证"""
    print("\n\n测试参数验证")
    print("=" * 50)
    
    mock_alpha_results = {
        f"layer_{i}": {
            'alpha': 1.0 - i * 0.1,
            'category': 'linear',
            'out_features': 512,
            'in_features': 512,
            'numel': 262144,
            'dtype': 'float32',
            'device': 'cpu'
        }
        for i in range(5)
    }
    
    # 测试无效的比例组合
    invalid_configs = [
        {"mxfp4_ratio": 0.6, "bf16_ratio": 0.5},  # 总和 > 1.0
        {"mxfp4_ratio": 0.8, "bf16_ratio": 0.3},  # 总和 > 1.0
    ]
    
    for i, config in enumerate(invalid_configs):
        print(f"\n测试无效配置 {i+1}: mxfp4_ratio={config['mxfp4_ratio']:.1f}, bf16_ratio={config['bf16_ratio']:.1f}")
        print("-" * 60)
        
        try:
            result = create_quantization_config(
                mock_alpha_results,
                mxfp4_ratio=config['mxfp4_ratio'],
                bf16_ratio=config['bf16_ratio']
            )
            print("意外成功，应该失败")
        except ValueError as e:
            print(f"正确捕获错误: {e}")


if __name__ == "__main__":
    test_three_tier_quantization()
    test_validation() 