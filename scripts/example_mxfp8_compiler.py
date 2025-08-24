#!/usr/bin/env python3
"""
使用 torch.compiler 加速的 mxfp8_torch 函数示例
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import time
from alphaquant.quantizers.kernel.fp_torch import (
    mxfp8_torch, 
    mxfp8_torch_compiled, 
    mxfp8_torch_compiled_optimized
)


def simple_benchmark():
    """简单的性能对比"""
    print("MXFP8 量化函数性能对比")
    print("=" * 40)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    
    # 创建测试数据
    x = torch.randn(512, 512, device=device)
    print(f"输入张量形状: {x.shape}")
    
    # 测试 e4m3 格式
    print("\n测试 e4m3 格式:")
    
    # 原始函数
    start_time = time.time()
    result_orig = mxfp8_torch(x, scaled_value_format='e4m3')
    orig_time = time.time() - start_time
    print(f"原始函数: {orig_time*1000:.3f} ms")
    
    # 编译版本
    start_time = time.time()
    result_compiled = mxfp8_torch_compiled(x, scaled_value_format='e4m3')
    compiled_time = time.time() - start_time
    print(f"编译版本: {compiled_time*1000:.3f} ms")
    
    # 优化版本
    start_time = time.time()
    result_optimized = mxfp8_torch_compiled_optimized(x, scaled_value_format='e4m3')
    optimized_time = time.time() - start_time
    print(f"优化版本: {optimized_time*1000:.3f} ms")
    
    # 计算加速比
    speedup_compiled = orig_time / compiled_time
    speedup_optimized = orig_time / optimized_time
    print(f"\n加速比:")
    print(f"编译版本: {speedup_compiled:.2f}x")
    print(f"优化版本: {speedup_optimized:.2f}x")
    
    # 验证结果一致性
    print(f"\n结果一致性检查:")
    print(f"原始 vs 编译: {torch.allclose(result_orig, result_compiled, atol=1e-6)}")
    print(f"原始 vs 优化: {torch.allclose(result_orig, result_optimized, atol=1e-6)}")
    
    return result_orig, result_compiled, result_optimized


def test_different_formats():
    """测试不同格式"""
    print("\n\n测试不同格式")
    print("=" * 40)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(256, 256, device=device)
    
    formats = ['e4m3', 'e5m2']
    
    for fmt in formats:
        print(f"\n格式: {fmt}")
        
        # 原始函数
        start_time = time.time()
        result_orig = mxfp8_torch(x, scaled_value_format=fmt)
        orig_time = time.time() - start_time
        
        # 编译版本
        start_time = time.time()
        result_compiled = mxfp8_torch_compiled(x, scaled_value_format=fmt)
        compiled_time = time.time() - start_time
        
        speedup = orig_time / compiled_time
        print(f"  加速比: {speedup:.2f}x")
        print(f"  结果一致: {torch.allclose(result_orig, result_compiled, atol=1e-6)}")


def test_different_sizes():
    """测试不同张量大小"""
    print("\n\n测试不同张量大小")
    print("=" * 40)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sizes = [64, 128, 256, 512, 1024]
    
    for size in sizes:
        x = torch.randn(size, size, device=device)
        print(f"\n张量大小: {size}x{size}")
        
        # 原始函数
        start_time = time.time()
        result_orig = mxfp8_torch(x, scaled_value_format='e4m3')
        orig_time = time.time() - start_time
        
        # 编译版本
        start_time = time.time()
        result_compiled = mxfp8_torch_compiled(x, scaled_value_format='e4m3')
        compiled_time = time.time() - start_time
        
        speedup = orig_time / compiled_time
        print(f"  加速比: {speedup:.2f}x")
        print(f"  结果一致: {torch.allclose(result_orig, result_compiled, atol=1e-6)}")


if __name__ == "__main__":
    print("torch.compiler 加速的 MXFP8 量化函数示例")
    print("=" * 60)
    
    try:
        # 基本性能对比
        simple_benchmark()
        
        # 测试不同格式
        test_different_formats()
        
        # 测试不同大小
        test_different_sizes()
        
    except Exception as e:
        print(f"运行出错: {e}")
        import traceback
        traceback.print_exc() 