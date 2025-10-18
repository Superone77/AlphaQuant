#!/usr/bin/env python3
"""
测试 torch.compiler 加速的 mxfp8_torch 函数性能
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import time
import numpy as np
from alphaquant.quantizers.kernel.fp_torch import (
    mxfp8_torch, 
    mxfp8_torch_compiled, 
    mxfp8_torch_compiled_optimized,
    mxfp8_torch_batch_optimized
)


def benchmark_function(func, x, num_runs=100, warmup_runs=10):
    """基准测试函数性能"""
    # 预热
    for _ in range(warmup_runs):
        _ = func(x)
    
    # 同步 GPU
    if x.is_cuda:
        torch.cuda.synchronize()
    
    # 计时
    start_time = time.time()
    for _ in range(num_runs):
        result = func(x)
        if x.is_cuda:
            torch.cuda.synchronize()
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs
    return avg_time, result


def test_accuracy(original_func, optimized_func, x):
    """测试优化函数的准确性"""
    with torch.no_grad():
        original_result = original_func(x)
        optimized_result = optimized_func(x)
        
        # 计算相对误差
        if torch.allclose(original_result, optimized_result, atol=1e-6, rtol=1e-6):
            return True, 0.0
        else:
            max_diff = torch.max(torch.abs(original_result - optimized_result))
            return False, max_diff.item()


def main():
    print("测试 torch.compiler 加速的 mxfp8_torch 函数")
    print("=" * 60)
    
    # 测试设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 测试数据
    test_shapes = [
        (32, 32),      # 小张量
        (128, 128),    # 中等张量
        (512, 512),    # 大张量
        (1024, 1024), # 超大张量
    ]
    
    for shape in test_shapes:
        print(f"\n测试张量形状: {shape}")
        print("-" * 40)
        
        # 生成测试数据
        x = torch.randn(*shape, device=device)
        
        # 测试不同格式
        for format in ['e4m3', 'e5m2']:
            print(f"格式: {format}")
            
            # 基准测试原始函数
            try:
                orig_time, orig_result = benchmark_function(
                    lambda t: mxfp8_torch(t, scaled_value_format=format), 
                    x, num_runs=50
                )
                print(f"  原始函数: {orig_time*1000:.3f} ms")
            except Exception as e:
                print(f"  原始函数错误: {e}")
                continue
            
            # 测试编译版本
            try:
                compiled_time, compiled_result = benchmark_function(
                    lambda t: mxfp8_torch_compiled(t, scaled_value_format=format), 
                    x, num_runs=50
                )
                speedup = orig_time / compiled_time
                print(f"  编译版本: {compiled_time*1000:.3f} ms (加速比: {speedup:.2f}x)")
                
                # 测试准确性
                is_correct, max_diff = test_accuracy(
                    lambda t: mxfp8_torch(t, scaled_value_format=format),
                    lambda t: mxfp8_torch_compiled(t, scaled_value_format=format),
                    x
                )
                print(f"  准确性: {'✓' if is_correct else '✗'} (最大差异: {max_diff:.6f})")
                
            except Exception as e:
                print(f"  编译版本错误: {e}")
            
            # 测试优化版本
            try:
                opt_time, opt_result = benchmark_function(
                    lambda t: mxfp8_torch_compiled_optimized(t, scaled_value_format=format), 
                    x, num_runs=50
                )
                speedup = orig_time / opt_time
                print(f"  优化版本: {opt_time*1000:.3f} ms (加速比: {speedup:.2f}x)")
                
                # 测试准确性
                is_correct, max_diff = test_accuracy(
                    lambda t: mxfp8_torch(t, scaled_value_format=format),
                    lambda t: mxfp8_torch_compiled_optimized(t, scaled_value_format=format),
                    x
                )
                print(f"  准确性: {'✓' if is_correct else '✗'} (最大差异: {max_diff:.6f})")
                
            except Exception as e:
                print(f"  优化版本错误: {e}")
            
            # 测试批量版本
            try:
                batch_time, batch_result = benchmark_function(
                    lambda t: mxfp8_torch_batch_optimized(t, scaled_value_format=format), 
                    x, num_runs=50
                )
                speedup = orig_time / batch_time
                print(f"  批量版本: {batch_time*1000:.3f} ms (加速比: {speedup:.2f}x)")
                
                # 测试准确性
                is_correct, max_diff = test_accuracy(
                    lambda t: mxfp8_torch(t, scaled_value_format=format),
                    lambda t: mxfp8_torch_batch_optimized(t, scaled_value_format=format),
                    x
                )
                print(f"  准确性: {'✓' if is_correct else '✗'} (最大差异: {max_diff:.6f})")
                
            except Exception as e:
                print(f"  批量版本错误: {e}")


if __name__ == "__main__":
    main() 