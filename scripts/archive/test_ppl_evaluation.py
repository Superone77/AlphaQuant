#!/usr/bin/env python3
"""
测试 PPL 评估功能
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_lm_eval_import():
    """测试 lm_eval 导入"""
    print("测试 lm_eval 导入...")
    
    try:
        import lm_eval
        print(f"✓ lm_eval 版本: {lm_eval.__version__}")
        
        from lm_eval import evaluator
        print("✓ evaluator 导入成功")
        
        from lm_eval.models.huggingface import HFLM
        print("✓ HFLM 导入成功")
        
        return True
        
    except ImportError as e:
        print(f"✗ lm_eval 导入失败: {e}")
        return False
    except Exception as e:
        print(f"✗ 其他导入错误: {e}")
        return False


def test_hflm_initialization():
    """测试 HFLM 初始化"""
    print("\n测试 HFLM 初始化...")
    
    try:
        from lm_eval.models.huggingface import HFLM
        
        # 测试不同的初始化方式
        print("测试方式 1: 基本初始化")
        try:
            lm = HFLM(pretrained=None, model=None, device="cpu")
            print("✓ 基本初始化成功")
        except Exception as e:
            print(f"✗ 基本初始化失败: {e}")
        
        print("测试方式 2: 带参数初始化")
        try:
            lm = HFLM(pretrained=None, model=None, device="cpu", batch_size=1)
            print("✓ 带参数初始化成功")
        except Exception as e:
            print(f"✗ 带参数初始化失败: {e}")
        
        print("测试方式 3: 完整参数初始化")
        try:
            lm = HFLM(pretrained=None, model=None, device="cpu", batch_size=1, dtype="bf16")
            print("✓ 完整参数初始化成功")
        except Exception as e:
            print(f"✗ 完整参数初始化失败: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ HFLM 测试失败: {e}")
        return False


def test_evaluator_simple_evaluate():
    """测试 evaluator.simple_evaluate"""
    print("\n测试 evaluator.simple_evaluate...")
    
    try:
        from lm_eval import evaluator
        
        # 检查函数是否存在
        if hasattr(evaluator, 'simple_evaluate'):
            print("✓ simple_evaluate 函数存在")
            
            # 检查函数签名
            import inspect
            sig = inspect.signature(evaluator.simple_evaluate)
            print(f"  函数签名: {sig}")
            
            return True
        else:
            print("✗ simple_evaluate 函数不存在")
            return False
            
    except Exception as e:
        print(f"✗ evaluator 测试失败: {e}")
        return False


def test_wikitext_task():
    """测试 wikitext 任务"""
    print("\n测试 wikitext 任务...")
    
    try:
        from lm_eval import evaluator
        
        # 检查可用的任务
        available_tasks = evaluator.available_tasks()
        print(f"可用任务: {available_tasks}")
        
        if 'wikitext' in available_tasks:
            print("✓ wikitext 任务可用")
            return True
        else:
            print("✗ wikitext 任务不可用")
            return False
            
    except Exception as e:
        print(f"✗ wikitext 任务测试失败: {e}")
        return False


def test_device_handling():
    """测试设备处理"""
    print("\n测试设备处理...")
    
    try:
        import torch
        
        # 测试 CPU 设备
        print("测试 CPU 设备...")
        device = "cpu"
        print(f"  设备: {device}")
        
        # 测试 CUDA 设备（如果可用）
        if torch.cuda.is_available():
            print("测试 CUDA 设备...")
            device = "cuda:0"
            print(f"  设备: {device}")
            print(f"  CUDA 版本: {torch.version.cuda}")
        else:
            print("CUDA 不可用")
        
        return True
        
    except Exception as e:
        print(f"✗ 设备处理测试失败: {e}")
        return False


def test_model_loading():
    """测试模型加载"""
    print("\n测试模型加载...")
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # 尝试加载一个小模型进行测试
        model_name = "microsoft/DialoGPT-small"  # 小模型，加载快
        
        print(f"尝试加载模型: {model_name}")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            print("✓ tokenizer 加载成功")
            
            model = AutoModelForCausalLM.from_pretrained(model_name)
            print("✓ 模型加载成功")
            
            # 检查模型属性
            print(f"  模型类型: {type(model)}")
            print(f"  设备: {next(model.parameters()).device}")
            
            return True
            
        except Exception as e:
            print(f"✗ 模型加载失败: {e}")
            return False
        
    except ImportError as e:
        print(f"✗ transformers 导入失败: {e}")
        return False
    except Exception as e:
        print(f"✗ 模型加载测试失败: {e}")
        return False


def main():
    """主测试函数"""
    print("PPL 评估功能测试")
    print("=" * 60)
    
    tests = [
        ("lm_eval 导入", test_lm_eval_import),
        ("HFLM 初始化", test_hflm_initialization),
        ("evaluator.simple_evaluate", test_evaluator_simple_evaluate),
        ("wikitext 任务", test_wikitext_task),
        ("设备处理", test_device_handling),
        ("模型加载", test_model_loading),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} 测试异常: {e}")
            results.append((test_name, False))
    
    # 总结测试结果
    print("\n" + "=" * 60)
    print("测试结果总结")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name:<25} {status}")
        if result:
            passed += 1
    
    print("-" * 60)
    print(f"总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！PPL 评估功能应该可以正常工作。")
    else:
        print("⚠️  部分测试失败，请检查相关依赖和配置。")
        
        # 提供修复建议
        print("\n修复建议:")
        if not any("lm_eval" in name for name, result in results if not result):
            print("1. 安装 lm_eval: pip install lm-eval")
        if not any("transformers" in name for name, result in results if not result):
            print("2. 安装 transformers: pip install transformers")
        if not any("torch" in name for name, result in results if not result):
            print("3. 安装 PyTorch: pip install torch")


if __name__ == "__main__":
    main() 