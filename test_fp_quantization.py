#!/usr/bin/env python
"""
测试 FP4 和 FP8 量化的实际效果
"""
import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from alphaquant.quantizers.kernel.int_torch import fake_quant_fp4, fake_quant_fp8


def test_quantization():
    """测试量化效果"""
    print("\n" + "="*60)
    print("测试 FP4 和 FP8 量化")
    print("="*60 + "\n")
    
    # 创建测试数据
    torch.manual_seed(42)
    x = torch.randn(10, 100) * 5.0
    
    print(f"原始数据统计:")
    print(f"  Shape: {x.shape}")
    print(f"  Min: {x.min():.6f}, Max: {x.max():.6f}")
    print(f"  Mean: {x.mean():.6f}, Std: {x.std():.6f}")
    print(f"  唯一值数量: {torch.unique(x).numel()}")
    
    # FP4 量化
    print(f"\n{'='*60}")
    print("FP4 E2M1 量化:")
    print('='*60)
    x_fp4 = fake_quant_fp4(x, format='e2m1')
    
    print(f"  Min: {x_fp4.min():.6f}, Max: {x_fp4.max():.6f}")
    print(f"  Mean: {x_fp4.mean():.6f}, Std: {x_fp4.std():.6f}")
    print(f"  唯一值数量: {torch.unique(x_fp4).numel()}")
    print(f"  MSE loss: {torch.mean((x - x_fp4)**2):.6f}")
    print(f"  前10个值: {x_fp4[0, :10].tolist()}")
    
    # FP8 量化
    print(f"\n{'='*60}")
    print("FP8 E4M3 量化:")
    print('='*60)
    x_fp8 = fake_quant_fp8(x, format='e4m3')
    
    print(f"  Min: {x_fp8.min():.6f}, Max: {x_fp8.max():.6f}")
    print(f"  Mean: {x_fp8.mean():.6f}, Std: {x_fp8.std():.6f}")
    print(f"  唯一值数量: {torch.unique(x_fp8).numel()}")
    print(f"  MSE loss: {torch.mean((x - x_fp8)**2):.6f}")
    print(f"  前10个值: {x_fp8[0, :10].tolist()}")
    
    # 对比
    print(f"\n{'='*60}")
    print("对比分析:")
    print('='*60)
    
    unique_fp4 = torch.unique(x_fp4).numel()
    unique_fp8 = torch.unique(x_fp8).numel()
    mse_fp4 = torch.mean((x - x_fp4)**2).item()
    mse_fp8 = torch.mean((x - x_fp8)**2).item()
    
    print(f"唯一值数量:")
    print(f"  FP4: {unique_fp4} 个")
    print(f"  FP8: {unique_fp8} 个")
    print(f"  比例: {unique_fp8/max(unique_fp4, 1):.2f}x")
    
    print(f"\nMSE 误差:")
    print(f"  FP4: {mse_fp4:.6f}")
    print(f"  FP8: {mse_fp8:.6f}")
    print(f"  FP4 误差是 FP8 的 {mse_fp4/max(mse_fp8, 1e-10):.2f}x")
    
    # 验证理论值
    print(f"\n{'='*60}")
    print("理论验证:")
    print('='*60)
    print(f"FP4 E2M1 理论可表示值: 8 (正数) × 2 (符号) = 16 个离散值")
    print(f"FP8 E4M3 理论可表示值: ~120 个离散值")
    print(f"实际 FP4 唯一值: {unique_fp4} ✓" if unique_fp4 < 30 else f"实际 FP4 唯一值: {unique_fp4} ✗")
    print(f"实际 FP8 唯一值: {unique_fp8} ✓" if unique_fp8 > 50 else f"实际 FP8 唯一值: {unique_fp8} ✗")
    
    if unique_fp4 < 30 and unique_fp8 > 50:
        print(f"\n✅ 测试通过！FP4 和 FP8 现在有明显区别")
    else:
        print(f"\n⚠️  警告：量化效果可能不符合预期")
    
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    test_quantization()

