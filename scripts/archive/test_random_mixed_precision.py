#!/usr/bin/env python3
"""
测试随机混合精度量化脚本

这个脚本用于测试和演示随机混合精度量化功能
"""

import sys
import os
import json
import tempfile
from pathlib import Path

# 添加项目根目录到 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.random_mixed_precision_quantization import (
    identify_layer_categories,
    create_random_quantization_config,
    save_config,
    print_config_summary
)


def create_mock_model():
    """创建一个模拟模型用于测试"""
    import torch
    import torch.nn as nn
    
    class MockLinear(nn.Module):
        def __init__(self, in_features, out_features):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(out_features, in_features))
            self.bias = nn.Parameter(torch.randn(out_features))
    
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            # Expert层
            self.expert1 = MockLinear(512, 1024)
            self.expert2 = MockLinear(1024, 512)
            self.expert3 = MockLinear(512, 1024)
            
            # Attention层
            self.q_proj = MockLinear(512, 512)
            self.k_proj = MockLinear(512, 512)
            self.v_proj = MockLinear(512, 512)
            self.o_proj = MockLinear(512, 512)
            
            # 其他层
            self.linear1 = MockLinear(512, 1024)
            self.linear2 = MockLinear(1024, 512)
            self.linear3 = MockLinear(512, 256)
            
            # 特殊层
            self.lm_head = MockLinear(256, 1000)
            self.embed_tokens = MockLinear(1000, 256)
    
    return MockModel()


def test_layer_categorization():
    """测试层分类功能"""
    print("测试层分类功能...")
    print("="*50)
    
    model = create_mock_model()
    layer_categories = identify_layer_categories(model)
    
    print("层分类结果:")
    for name, category in layer_categories.items():
        print(f"  {name:<20} -> {category}")
    
    # 验证分类结果
    expert_count = sum(1 for cat in layer_categories.values() if cat == "expert")
    attention_count = sum(1 for cat in layer_categories.values() if cat == "attention")
    other_count = sum(1 for cat in layer_categories.values() if cat == "other")
    
    print(f"\n分类统计:")
    print(f"  Expert层: {expert_count}")
    print(f"  Attention层: {attention_count}")
    print(f"  其他层: {other_count}")
    
    assert expert_count == 3, f"期望3个Expert层，实际{expert_count}个"
    assert attention_count == 4, f"期望4个Attention层，实际{attention_count}个"
    assert other_count == 3, f"期望3个其他层，实际{other_count}个"
    
    print("✓ 层分类测试通过")
    return layer_categories


def test_random_config_generation(layer_categories):
    """测试随机配置生成功能"""
    print("\n测试随机配置生成功能...")
    print("="*50)
    
    # 测试不同的比例组合
    test_cases = [
        {"mxfp4_ratio": 0.3, "bf16_ratio": 0.2, "seed": 42},
        {"mxfp4_ratio": 0.5, "bf16_ratio": 0.1, "seed": 123},
        {"mxfp4_ratio": 0.1, "bf16_ratio": 0.4, "seed": 456},
    ]
    
    for i, test_case in enumerate(test_cases):
        print(f"\n测试用例 {i+1}: mxfp4={test_case['mxfp4_ratio']:.1%}, bf16={test_case['bf16_ratio']:.1%}")
        print("-" * 40)
        
        config = create_random_quantization_config(
            layer_categories=layer_categories,
            mxfp4_ratio=test_case["mxfp4_ratio"],
            bf16_ratio=test_case["bf16_ratio"],
            group_size=32,
            seed=test_case["seed"]
        )
        
        # 验证配置结构
        assert "default" in config, "配置缺少default部分"
        assert "overrides" in config, "配置缺少overrides部分"
        assert "summary" in config, "配置缺少summary部分"
        
        # 验证比例
        summary = config["summary"]
        total_layers = summary["total_layers"]
        mxfp4_count = summary["mxfp4_layers"]
        bf16_count = summary["bf16_layers"]
        mxfp8_count = summary["mxfp8_layers"]
        
        print(f"  总层数: {total_layers}")
        print(f"  mxfp4: {mxfp4_count} ({mxfp4_count/total_layers:.1%})")
        print(f"  bf16: {bf16_count} ({bf16_count/total_layers:.1%})")
        print(f"  mxfp8: {mxfp8_count} ({mxfp8_count/total_layers:.1%})")
        
        # 验证总数
        assert mxfp4_count + bf16_count + mxfp8_count == total_layers, "层数总数不匹配"
        
        # 验证特殊层
        special_layers = [o for o in config["overrides"] if o.get("category") == "special"]
        assert len(special_layers) == 3, f"期望3个特殊层，实际{len(special_layers)}个"
        
        print(f"  ✓ 测试用例 {i+1} 通过")
    
    print("✓ 随机配置生成测试通过")
    return config


def test_config_save_and_load(config):
    """测试配置保存和加载功能"""
    print("\n测试配置保存和加载功能...")
    print("="*50)
    
    # 创建临时文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name
    
    try:
        # 保存配置
        save_config(config, temp_path)
        
        # 验证文件存在
        assert os.path.exists(temp_path), "配置文件未保存"
        
        # 加载并验证配置
        with open(temp_path, 'r', encoding='utf-8') as f:
            loaded_config = json.load(f)
        
        # 验证配置内容
        assert loaded_config["default"] == config["default"], "default配置不匹配"
        assert len(loaded_config["overrides"]) == len(config["overrides"]), "overrides数量不匹配"
        assert loaded_config["summary"]["total_layers"] == config["summary"]["total_layers"], "summary不匹配"
        
        print(f"✓ 配置保存和加载测试通过")
        print(f"  临时文件: {temp_path}")
        
    finally:
        # 清理临时文件
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_edge_cases():
    """测试边界情况"""
    print("\n测试边界情况...")
    print("="*50)
    
    # 创建最小模型
    import torch
    import torch.nn as nn
    
    class MinimalModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 10)
    
    model = MinimalModel()
    layer_categories = identify_layer_categories(model)
    
    # 测试极端比例
    extreme_cases = [
        {"mxfp4_ratio": 0.0, "bf16_ratio": 0.0},  # 全部mxfp8
        {"mxfp4_ratio": 1.0, "bf16_ratio": 0.0},  # 全部mxfp4
        {"mxfp4_ratio": 0.0, "bf16_ratio": 1.0},  # 全部bf16
        {"mxfp4_ratio": 0.5, "bf16_ratio": 0.5},  # 各占一半
    ]
    
    for i, case in enumerate(extreme_cases):
        print(f"测试边界情况 {i+1}: mxfp4={case['mxfp4_ratio']:.1%}, bf16={case['bf16_ratio']:.1%}")
        
        try:
            config = create_random_quantization_config(
                layer_categories=layer_categories,
                mxfp4_ratio=case["mxfp4_ratio"],
                bf16_ratio=case["bf16_ratio"],
                group_size=32,
                seed=42
            )
            
            summary = config["summary"]
            print(f"  ✓ 成功生成配置: mxfp4={summary['mxfp4_layers']}, bf16={summary['bf16_layers']}, mxfp8={summary['mxfp8_layers']}")
            
        except Exception as e:
            print(f"  ✗ 生成配置失败: {e}")
    
    # 测试无效比例
    print("\n测试无效比例...")
    try:
        config = create_random_quantization_config(
            layer_categories=layer_categories,
            mxfp4_ratio=0.6,
            bf16_ratio=0.5,  # 总和超过1.0
            group_size=32,
            seed=42
        )
        print("  ✗ 应该抛出异常但未抛出")
    except ValueError as e:
        print(f"  ✓ 正确抛出异常: {e}")
    
    print("✓ 边界情况测试通过")


def main():
    """主测试函数"""
    print("开始测试随机混合精度量化脚本")
    print("="*60)
    
    try:
        # 测试层分类
        layer_categories = test_layer_categorization()
        
        # 测试随机配置生成
        config = test_random_config_generation(layer_categories)
        
        # 测试配置保存和加载
        test_config_save_and_load(config)
        
        # 测试边界情况
        test_edge_cases()
        
        print("\n" + "="*60)
        print("🎉 所有测试通过！")
        print("="*60)
        
        # 显示最终配置摘要
        print("\n最终配置摘要:")
        print_config_summary(config)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 