#!/usr/bin/env python3
"""
测试新实现的MXFP4和MXFP8 triton kernels
"""

import torch
import sys
import os

# 添加项目根目录到sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from alphaquant.quantizers.kernel.mxfp4_triton import mxfp4_forward
from alphaquant.quantizers.kernel.mxfp8_triton import mxfp8_forward, mxfp8_e4m3_forward, mxfp8_e5m2_forward
from alphaquant.quantizers.kernel.nvfp4_triton import nvfp4_forward

def test_kernels():
    """测试所有triton kernels"""
    print("Testing Triton Kernels...")
    
    # 检查CUDA是否可用
    if not torch.cuda.is_available():
        print("CUDA not available, skipping GPU tests")
        return
    
    device = torch.device('cuda')
    torch.manual_seed(42)
    
    # 测试数据
    B, N = 2, 32
    x = torch.randn(B, N, device=device, dtype=torch.float16) * 10
    
    print(f"Input shape: {x.shape}")
    print(f"Input range: [{x.min().item():.4f}, {x.max().item():.4f}]")
    print(f"Input sample: {x[0, :8]}")
    print()
    
    try:
        # 测试NVFP4
        print("Testing NVFP4...")
        y_nvfp4_det = nvfp4_forward(x, stochastic_rounding=False)
        y_nvfp4_sto = nvfp4_forward(x, stochastic_rounding=True)
        print(f"NVFP4 det: {y_nvfp4_det[0, :8]}")
        print(f"NVFP4 sto: {y_nvfp4_sto[0, :8]}")
        print()
        
        # 测试MXFP4
        print("Testing MXFP4...")
        y_mxfp4_det = mxfp4_forward(x, stochastic_rounding=False)
        y_mxfp4_sto = mxfp4_forward(x, stochastic_rounding=True)
        print(f"MXFP4 det: {y_mxfp4_det[0, :8]}")
        print(f"MXFP4 sto: {y_mxfp4_sto[0, :8]}")
        print()
        
        # 测试MXFP8 E4M3
        print("Testing MXFP8 E4M3...")
        y_mxfp8_e4m3_det = mxfp8_e4m3_forward(x, stochastic_rounding=False)
        y_mxfp8_e4m3_sto = mxfp8_e4m3_forward(x, stochastic_rounding=True)
        print(f"MXFP8 E4M3 det: {y_mxfp8_e4m3_det[0, :8]}")
        print(f"MXFP8 E4M3 sto: {y_mxfp8_e4m3_sto[0, :8]}")
        print()
        
        # 测试MXFP8 E5M2
        print("Testing MXFP8 E5M2...")
        y_mxfp8_e5m2_det = mxfp8_e5m2_forward(x, stochastic_rounding=False)
        y_mxfp8_e5m2_sto = mxfp8_e5m2_forward(x, stochastic_rounding=True)
        print(f"MXFP8 E5M2 det: {y_mxfp8_e5m2_det[0, :8]}")
        print(f"MXFP8 E5M2 sto: {y_mxfp8_e5m2_sto[0, :8]}")
        print()
        
        # 测试通用MXFP8接口
        print("Testing generic MXFP8 interface...")
        y_mxfp8_gen_e4m3 = mxfp8_forward(x, format="e4m3", stochastic_rounding=False)
        y_mxfp8_gen_e5m2 = mxfp8_forward(x, format="e5m2", stochastic_rounding=False)
        print(f"MXFP8 generic E4M3: {y_mxfp8_gen_e4m3[0, :8]}")
        print(f"MXFP8 generic E5M2: {y_mxfp8_gen_e5m2[0, :8]}")
        print()
        
        print("✅ All kernels tested successfully!")
        
    except Exception as e:
        print(f"❌ Error testing kernels: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_kernels() 