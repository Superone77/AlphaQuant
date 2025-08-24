#!/usr/bin/env python3
"""
对比NVFP4和MXFP4/MXFP8的差异
"""

import torch
import sys
import os

# 添加项目根目录到sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from alphaquant.quantizers.kernel.nvfp4_triton import nvfp4_forward
from alphaquant.quantizers.kernel.mxfp4_triton import mxfp4_forward
from alphaquant.quantizers.kernel.mxfp8_triton import mxfp8_forward

def compare_kernels():
    """对比不同kernels的行为"""
    print("Comparing NVFP4 vs MXFP4/MXFP8 Kernels...")
    
    # 检查CUDA是否可用
    if not torch.cuda.is_available():
        print("CUDA not available, skipping GPU tests")
        return
    
    device = torch.device('cuda')
    torch.manual_seed(42)
    
    # 测试数据 - 使用较大的值来观察scaling差异
    B, N = 2, 32
    x = torch.randn(B, N, device=device, dtype=torch.float16) * 100
    
    print(f"Input shape: {x.shape}")
    print(f"Input range: [{x.min().item():.4f}, {x.max().item():.4f}]")
    print(f"Input sample: {x[0, :8]}")
    print()
    
    try:
        # 测试NVFP4 (有global scaling)
        print("Testing NVFP4 (with global scaling)...")
        y_nvfp4 = nvfp4_forward(x, stochastic_rounding=False)
        print(f"NVFP4 output: {y_nvfp4[0, :8]}")
        print(f"NVFP4 range: [{y_nvfp4.min().item():.4f}, {y_nvfp4.max().item():.4f}]")
        print()
        
        # 测试MXFP4 (无global scaling)
        print("Testing MXFP4 (without global scaling)...")
        y_mxfp4 = mxfp4_forward(x, stochastic_rounding=False)
        print(f"MXFP4 output: {y_mxfp4[0, :8]}")
        print(f"MXFP4 range: [{y_mxfp4.min().item():.4f}, {y_mxfp4.max().item():.4f}]")
        print()
        
        # 测试MXFP8 E4M3 (无global scaling)
        print("Testing MXFP8 E4M3 (without global scaling)...")
        y_mxfp8_e4m3 = mxfp8_forward(x, format="e4m3", stochastic_rounding=False)
        print(f"MXFP8 E4M3 output: {y_mxfp8_e4m3[0, :8]}")
        print(f"MXFP8 E4M3 range: [{y_mxfp8_e4m3.min().item():.4f}, {y_mxfp8_e4m3.max().item():.4f}]")
        print()
        
        # 测试MXFP8 E5M2 (无global scaling)
        print("Testing MXFP8 E5M2 (without global scaling)...")
        y_mxfp8_e5m2 = mxfp8_forward(x, format="e5m2", stochastic_rounding=False)
        print(f"MXFP8 E5M2 output: {y_mxfp8_e5m2[0, :8]}")
        print(f"MXFP8 E5M2 range: [{y_mxfp8_e5m2.min().item():.4f}, {y_mxfp8_e5m2.max().item():.4f}]")
        print()
        
        # 分析差异
        print("=== 差异分析 ===")
        print(f"NVFP4 vs MXFP4 输出差异: {torch.abs(y_nvfp4 - y_mxfp4).mean().item():.6f}")
        print(f"NVFP4 vs MXFP8 E4M3 输出差异: {torch.abs(y_nvfp4 - y_mxfp8_e4m3).mean().item():.6f}")
        print(f"MXFP4 vs MXFP8 E4M3 输出差异: {torch.abs(y_mxfp4 - y_mxfp8_e4m3).mean().item():.6f}")
        print()
        
        print("✅ Kernel comparison completed!")
        print("\n主要差异:")
        print("1. NVFP4使用global + per-block scaling，输出范围更小")
        print("2. MXFP4/MXFP8只使用per-block scaling，保持更多原始动态范围")
        print("3. Block size: NVFP4=16, MXFP4/MXFP8=32")
        
    except Exception as e:
        print(f"❌ Error comparing kernels: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    compare_kernels() 