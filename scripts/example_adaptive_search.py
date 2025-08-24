#!/usr/bin/env python3
"""
自适应量化搜索算法使用示例
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.adaptive_quantization_search import AdaptiveQuantizationSearch


def example_basic_usage():
    """基本使用示例"""
    print("示例 1: 基本使用")
    print("=" * 60)
    
    # 创建搜索器
    searcher = AdaptiveQuantizationSearch(
        model_path="meta-llama/Llama-3.2-1B",  # 示例模型
        device="cpu",                           # 使用 CPU 进行测试
        dtype="bf16",                           # 数据类型
        ppl_threshold=0.1,                      # 允许 10% 的 PPL 增加
        target_avg_bits=8.0,                    # 目标平均 8 bits
        step_size=0.5,                          # 搜索步长
        batch_size=1                            # 评估 batch size
    )
    
    print("搜索器配置:")
    print(f"  模型: {searcher.model_path}")
    print(f"  设备: {searcher.device}")
    print(f"  数据类型: {searcher.dtype}")
    print(f"  PPL 阈值: {searcher.ppl_threshold:.1%}")
    print(f"  目标平均 bit 数: {searcher.target_avg_bits}")
    print(f"  搜索步长: {searcher.step_size}")
    print(f"  评估 batch size: {searcher.batch_size}")
    
    print("\n注意: 这是一个示例，实际运行时需要:")
    print("  1. 有效的模型路径或 HuggingFace 模型 ID")
    print("  2. 足够的计算资源")
    print("  3. 安装必要的依赖包")


def example_parameter_tuning():
    """参数调优示例"""
    print("\n\n示例 2: 参数调优")
    print("=" * 60)
    
    # 不同的参数组合
    parameter_combinations = [
        {
            "name": "保守策略",
            "ppl_threshold": 0.05,      # 只允许 5% PPL 增加
            "target_avg_bits": 10.0,    # 目标平均 10 bits
            "step_size": 0.3            # 较小的搜索步长
        },
        {
            "name": "平衡策略",
            "ppl_threshold": 0.1,       # 允许 10% PPL 增加
            "target_avg_bits": 8.0,     # 目标平均 8 bits
            "step_size": 0.5            # 中等搜索步长
        },
        {
            "name": "激进策略",
            "ppl_threshold": 0.2,       # 允许 20% PPL 增加
            "target_avg_bits": 6.0,     # 目标平均 6 bits
            "step_size": 0.8            # 较大的搜索步长
        }
    ]
    
    print("不同参数组合的效果对比:")
    print("-" * 80)
    print(f"{'策略名称':<15} {'PPL阈值':<12} {'目标Bit数':<12} {'搜索步长':<12} {'预期效果':<20}")
    print("-" * 80)
    
    for combo in parameter_combinations:
        expected_effect = "高精度，低压缩" if combo["ppl_threshold"] < 0.1 else \
                         "平衡精度和压缩" if combo["ppl_threshold"] < 0.15 else \
                         "高压缩，精度可能下降"
        
        print(f"{combo['name']:<15} {combo['ppl_threshold']:<12.1%} "
              f"{combo['target_avg_bits']:<12.1f} {combo['step_size']:<12.1f} "
              f"{expected_effect:<20}")


def example_search_process():
    """搜索过程示例"""
    print("\n\n示例 3: 搜索过程演示")
    print("=" * 60)
    
    # 模拟搜索过程
    print("搜索过程分为两个阶段:")
    print()
    
    print("阶段 1: 确定 mxfp4 阈值")
    print("  - 从最大 alpha 值开始")
    print("  - 逐步降低阈值")
    print("  - 当 PPL 增加超过阈值时停止")
    print("  - 记录 mxfp4 阈值")
    print()
    
    print("阶段 2: 确定 mxfp8 阈值")
    print("  - 从最大 alpha 值开始")
    print("  - 逐步降低阈值")
    print("  - 当 PPL 增加超过阈值或达到最小 alpha 值时停止")
    print("  - 记录 mxfp8 阈值")
    print()
    
    print("搜索过程中的检查点:")
    print("  ✓ 每次更新阈值后重新计算平均 bit 数")
    print("  ✓ 达到目标平均 bit 数时提前停止")
    print("  ✓ 记录完整的搜索历史")
    print("  ✓ 保护 lm_head 和 mlp.gate 层不被量化")


def example_output_formats():
    """输出格式示例"""
    print("\n\n示例 4: 输出格式")
    print("=" * 60)
    
    print("搜索完成后会生成两种文件:")
    print()
    
    print("1. 量化配置文件 (JSON)")
    print("   - 包含所有层的量化设置")
    print("   - 支持通配符模式匹配")
    print("   - 包含搜索摘要信息")
    print()
    
    print("2. 量化计划文件 (JSON)")
    print("   - 可直接用于模型量化")
    print("   - 包含每层的具体配置")
    print("   - 支持跳过特定层")
    print()
    
    print("配置文件结构示例:")
    config_example = {
        "default": {"wq": "bf16", "aq": "bf16", "group_size": 32},
        "overrides": [
            {"pattern": "model.layers.0.*", "wq": "mxfp4", "aq": "mxfp4", "group_size": 32},
            {"pattern": "model.layers.1.*", "wq": "mxfp8", "aq": "mxfp8", "group_size": 32},
            {"pattern": "lm_head", "skip": True}
        ],
        "search_summary": {
            "mxfp4_threshold": 8.5,
            "mxfp8_threshold": 5.0,
            "ppl_threshold": 0.1,
            "target_avg_bits": 8.0
        }
    }
    
    print("   default: 默认量化配置")
    print("   overrides: 特殊层配置")
    print("   search_summary: 搜索过程摘要")


def example_usage_workflow():
    """使用工作流程示例"""
    print("\n\n示例 5: 完整使用工作流程")
    print("=" * 60)
    
    print("步骤 1: 准备环境")
    print("  - 安装必要的依赖包")
    print("  - 准备模型文件或 HuggingFace 模型 ID")
    print("  - 确保有足够的计算资源")
    print()
    
    print("步骤 2: 配置参数")
    print("  - 设置 PPL 增加阈值 (建议: 0.05-0.2)")
    print("  - 设置目标平均 bit 数 (建议: 6.0-12.0)")
    print("  - 设置搜索步长 (建议: 0.3-0.8)")
    print()
    
    print("步骤 3: 运行搜索")
    print("  - 执行搜索算法")
    print("  - 监控搜索进度")
    print("  - 等待搜索完成")
    print()
    
    print("步骤 4: 分析结果")
    print("  - 检查量化配置")
    print("  - 验证平均 bit 数")
    print("  - 评估 PPL 变化")
    print()
    
    print("步骤 5: 应用量化")
    print("  - 使用生成的配置文件")
    print("  - 应用量化到模型")
    print("  - 验证量化效果")


if __name__ == "__main__":
    print("自适应量化搜索算法使用示例")
    print("=" * 80)
    
    try:
        example_basic_usage()
        example_parameter_tuning()
        example_search_process()
        example_output_formats()
        example_usage_workflow()
        
        print("\n" + "=" * 80)
        print("所有示例演示完成！")
        print("\n要运行实际的搜索，请使用:")
        print("python scripts/adaptive_quantization_search.py \\")
        print("    --model your_model_id \\")
        print("    --ppl-threshold 0.1 \\")
        print("    --target-avg-bits 8.0 \\")
        print("    --output-config config.json \\")
        print("    --output-plan plan.json")
        
    except Exception as e:
        print(f"\n示例演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc() 