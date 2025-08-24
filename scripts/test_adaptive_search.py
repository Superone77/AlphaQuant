#!/usr/bin/env python3
"""
测试自适应量化搜索算法
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.adaptive_quantization_search import AdaptiveQuantizationSearch


def test_search_algorithm():
    """测试搜索算法"""
    print("测试自适应量化搜索算法")
    print("=" * 60)
    
    # 创建搜索器（使用模拟数据）
    searcher = AdaptiveQuantizationSearch(
        model_path="test_model",
        device="cpu",
        dtype="bf16",
        ppl_threshold=0.1,      # 10% PPL 增加阈值
        target_avg_bits=8.0,    # 目标平均 8 bits
        step_size=0.5,          # 搜索步长
        batch_size=1
    )
    
    # 模拟 Alpha_Hill 结果
    mock_alpha_results = {
        f"layer_{i}": {
            'alpha': 10.0 - i * 0.5,  # 从 10.0 递减到 0.5
            'category': 'linear',
            'out_features': 512,
            'in_features': 512,
            'numel': 262144,
            'dtype': 'float32',
            'device': 'cpu'
        }
        for i in range(20)
    }
    
    searcher.alpha_results = mock_alpha_results
    
    print("模拟的 Alpha_Hill 值:")
    for name, result in mock_alpha_results.items():
        print(f"  {name}: alpha={result['alpha']:.3f}")
    
    # 测试 Alpha_Hill 计算
    print("\n测试 Alpha_Hill 计算...")
    sorted_layers = searcher.compute_alpha_hill()
    print(f"排序后的层数: {len(sorted_layers)}")
    print("前5层:")
    for i, (name, result) in enumerate(sorted_layers[:5]):
        print(f"  {i+1}. {name}: alpha={result['alpha']:.3f}")
    
    # 测试配置创建
    print("\n测试配置创建...")
    
    # 测试 mxfp4 阈值搜索配置
    mxfp4_config = searcher.create_search_config(sorted_layers, 8.0, None)
    print(f"mxfp4 阈值 8.0 的配置层数: {len(mxfp4_config)}")
    
    # 测试 mxfp8 阈值搜索配置
    mxfp8_config = searcher.create_search_config(sorted_layers, 8.0, 5.0)
    print(f"mxfp4 阈值 8.0, mxfp8 阈值 5.0 的配置层数: {len(mxfp8_config)}")
    
    # 统计不同精度的层数
    def count_precision_layers(config):
        mxfp4_count = sum(1 for c in config.values() if c.get('wq') == 'mxfp4')
        mxfp8_count = sum(1 for c in config.values() if c.get('wq') == 'mxfp8')
        bf16_count = sum(1 for c in config.values() if c.get('wq') == 'bf16')
        return mxfp4_count, mxfp8_count, bf16_count
    
    mxfp4_count, mxfp8_count, bf16_count = count_precision_layers(mxfp4_config)
    print(f"mxfp4 配置 - mxfp4: {mxfp4_count}, mxfp8: {mxfp8_count}, bf16: {bf16_count}")
    
    mxfp4_count, mxfp8_count, bf16_count = count_precision_layers(mxfp8_config)
    print(f"mxfp8 配置 - mxfp4: {mxfp4_count}, mxfp8: {mxfp8_count}, bf16: {bf16_count}")
    
    # 测试最终配置创建
    print("\n测试最终配置创建...")
    final_config = searcher.create_final_config(8.0, 5.0)
    print(f"最终配置层数: {len(final_config['overrides'])}")
    
    # 显示配置摘要
    print("\n配置摘要:")
    print(f"mxfp4 阈值: {final_config['search_summary']['mxfp4_threshold']}")
    print(f"mxfp8 阈值: {final_config['search_summary']['mxfp8_threshold']}")
    print(f"PPL 阈值: {final_config['search_summary']['ppl_threshold']}")
    print(f"目标平均 bit 数: {final_config['search_summary']['target_avg_bits']}")


def test_search_logic():
    """测试搜索逻辑"""
    print("\n\n测试搜索逻辑")
    print("=" * 60)
    
    # 模拟搜索过程
    searcher = AdaptiveQuantizationSearch(
        model_path="test_model",
        ppl_threshold=0.1,
        target_avg_bits=8.0,
        step_size=0.5
    )
    
    # 模拟 Alpha_Hill 结果
    mock_alpha_results = {
        f"layer_{i}": {
            'alpha': 10.0 - i * 0.5,
            'category': 'linear',
            'out_features': 512,
            'in_features': 512,
            'numel': 262144,
            'dtype': 'float32',
            'device': 'cpu'
        }
        for i in range(20)
    }
    
    searcher.alpha_results = mock_alpha_results
    
    # 模拟搜索历史
    search_history = [
        {
            'stage': 'mxfp4_search',
            'threshold': 9.5,
            'avg_bits': 15.5,
            'ppl': 15.2,
            'ppl_increase': 0.05,
            'config': {}
        },
        {
            'stage': 'mxfp4_search',
            'threshold': 9.0,
            'avg_bits': 14.8,
            'ppl': 15.8,
            'ppl_increase': 0.08,
            'config': {}
        },
        {
            'stage': 'mxfp4_search',
            'threshold': 8.5,
            'avg_bits': 14.2,
            'ppl': 16.5,
            'ppl_increase': 0.12,
            'config': {}
        }
    ]
    
    searcher.search_history = search_history
    
    print("模拟搜索历史:")
    for i, record in enumerate(search_history):
        print(f"  步骤 {i+1}: 阈值={record['threshold']:.1f}, "
              f"平均bit={record['avg_bits']:.1f}, "
              f"PPL增加={record['ppl_increase']:.1%}")
    
    # 分析搜索过程
    print("\n搜索过程分析:")
    
    # 找到 PPL 超出阈值的步骤
    ppl_threshold = 0.1
    for record in search_history:
        if record['ppl_increase'] > ppl_threshold:
            print(f"  在阈值 {record['threshold']:.1f} 时 PPL 增加超出阈值 "
                  f"({record['ppl_increase']:.1%} > {ppl_threshold:.1%})")
            break
    
    # 找到达到目标 bit 数的步骤
    target_bits = 8.0
    for record in search_history:
        if record['avg_bits'] <= target_bits:
            print(f"  在阈值 {record['threshold']:.1f} 时达到目标 bit 数 "
                  f"({record['avg_bits']:.1f} <= {target_bits})")
            break
    else:
        print(f"  未达到目标 bit 数 {target_bits}")


def test_config_validation():
    """测试配置验证"""
    print("\n\n测试配置验证")
    print("=" * 60)
    
    # 测试不同配置的有效性
    test_configs = [
        {
            "name": "全 bf16 配置",
            "config": {
                "default": {"wq": "bf16", "aq": "bf16"},
                "overrides": []
            }
        },
        {
            "name": "混合精度配置",
            "config": {
                "default": {"wq": "mxfp8", "aq": "mxfp8"},
                "overrides": [
                    {"pattern": "layer_0", "wq": "mxfp4", "aq": "mxfp4"},
                    {"pattern": "lm_head", "skip": True}
                ]
            }
        },
        {
            "name": "跳过层配置",
            "config": {
                "default": {"wq": "mxfp8", "aq": "mxfp8"},
                "overrides": [
                    {"pattern": "lm_head", "skip": True},
                    {"pattern": "mlp.gate", "skip": True}
                ]
            }
        }
    ]
    
    for test_config in test_configs:
        print(f"\n{test_config['name']}:")
        config = test_config['config']
        
        # 检查默认配置
        if 'default' in config:
            default = config['default']
            print(f"  默认配置: wq={default.get('wq', 'N/A')}, aq={default.get('aq', 'N/A')}")
        
        # 检查覆盖配置
        if 'overrides' in config:
            overrides = config['overrides']
            print(f"  覆盖配置数量: {len(overrides)}")
            
            for override in overrides:
                if override.get('skip'):
                    print(f"    跳过: {override['pattern']}")
                else:
                    print(f"    量化: {override['pattern']} -> {override.get('wq', 'N/A')}")


if __name__ == "__main__":
    print("自适应量化搜索算法测试")
    print("=" * 80)
    
    try:
        test_search_algorithm()
        test_search_logic()
        test_config_validation()
        
        print("\n" + "=" * 80)
        print("所有测试完成！")
        
    except Exception as e:
        print(f"\n测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc() 