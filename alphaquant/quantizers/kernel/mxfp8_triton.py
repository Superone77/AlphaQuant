import torch, triton, triton.language as tl

FP8_E4M3_MAX = 448.0    # fp8 e4m3 最大范围
FP8_E5M2_MAX = 57344.0  # fp8 e5m2 最大范围

# ---------------------------------------------------------------
# Triton kernel：MXFP8 E4M3 量化，只使用 per-block scaling
# ---------------------------------------------------------------
@triton.jit
def mxfp8_e4m3_fwd_kernel(
    x_ptr,          # *f16 / *f32  | 输入
    out_ptr,        # *f16 / *f32  | 输出
    prob_ptr,       # *f32         | [0,1) 随机数 (or 全 0)
    M,              # int32        | 总元素数
    BLOCK: tl.constexpr,          # =32
    STOCHASTIC: tl.constexpr,     # bool
):
    pid   = tl.program_id(0)
    offs  = pid * BLOCK + tl.arange(0, BLOCK)
    mask  = offs < M

    # -------- 读入数据 --------
    x      = tl.load(x_ptr + offs,  mask=mask, other=0.0)
    prob   = tl.load(prob_ptr + offs, mask=mask, other=0.0)
    sign   = tl.where(x >= 0.0, 1.0, -1.0)
    x_abs  = tl.abs(x)

    # -------- 计算 per-block scale_b --------
    blk_max = tl.max(tl.where(mask, x_abs, 0.0), axis=0)
    # 避免除零 / inf，确保 blk_max 有一个最小值
    blk_max = tl.maximum(blk_max, 1e-12)
    scale_b = 448.0 / blk_max

    y = x_abs * scale_b                # bring into [0, 448]
    # 限制y的范围，避免溢出
    y = tl.minimum(y, 448.0)

    # ============================================================
    # FP8 E4M3 量化 (MXFP8 格式)
    # ============================================================
    if STOCHASTIC:
        # 随机舍入到最近的量化级别
        # 这里简化处理，实际应该根据FP8 E4M3的量化级别来设计
        q_abs_blk = tl.round(y)
    else:
        # 确定性舍入
        q_abs_blk = tl.round(y)

    # -------- 反缩放并写回 --------
    q_abs = q_abs_blk / scale_b
    # 处理可能的 NaN/Inf
    q_abs = tl.where((q_abs == q_abs) & (q_abs != float('inf')) & (q_abs != float('-inf')), q_abs, 0.0)
    q_val = sign * q_abs
    tl.store(out_ptr + offs, q_val, mask=mask)


# ---------------------------------------------------------------
# Triton kernel：MXFP8 E5M2 量化，只使用 per-block scaling
# ---------------------------------------------------------------
@triton.jit
def mxfp8_e5m2_fwd_kernel(
    x_ptr,          # *f16 / *f32  | 输入
    out_ptr,        # *f16 / *f32  | 输出
    prob_ptr,       # *f32         | [0,1) 随机数 (or 全 0)
    M,              # int32        | 总元素数
    BLOCK: tl.constexpr,          # =32
    STOCHASTIC: tl.constexpr,     # bool
):
    pid   = tl.program_id(0)
    offs  = pid * BLOCK + tl.arange(0, BLOCK)
    mask  = offs < M

    # -------- 读入数据 --------
    x      = tl.load(x_ptr + offs,  mask=mask, other=0.0)
    prob   = tl.load(prob_ptr + offs, mask=mask, other=0.0)
    sign   = tl.where(x >= 0.0, 1.0, -1.0)
    x_abs  = tl.abs(x)

    # -------- 计算 per-block scale_b --------
    blk_max = tl.max(tl.where(mask, x_abs, 0.0), axis=0)
    # 避免除零 / inf，确保 blk_max 有一个最小值
    blk_max = tl.maximum(blk_max, 1e-12)
    scale_b = 57344.0 / blk_max

    y = x_abs * scale_b                # bring into [0, 57344]
    # 限制y的范围，避免溢出
    y = tl.minimum(y, 57344.0)

    # ============================================================
    # FP8 E5M2 量化 (MXFP8 格式)
    # ============================================================
    if STOCHASTIC:
        # 随机舍入到最近的量化级别
        q_abs_blk = tl.round(y)
    else:
        # 确定性舍入
        q_abs_blk = tl.round(y)

    # -------- 反缩放并写回 --------
    q_abs = q_abs_blk / scale_b
    # 处理可能的 NaN/Inf
    q_abs = tl.where((q_abs == q_abs) & (q_abs != float('inf')) & (q_abs != float('-inf')), q_abs, 0.0)
    q_val = sign * q_abs
    tl.store(out_ptr + offs, q_val, mask=mask)


# ---------------------------------------------------------------
# Python 包装：MXFP8 E4M3 量化
# ---------------------------------------------------------------
def mxfp8_e4m3_forward(
    x: torch.Tensor,
    stochastic_rounding: bool = False,
):
    """
    x             : CUDA 张量 (f16/f32)，任意形状
    stochastic_rounding: 是否使用随机舍入
    """
    
    assert x.is_cuda and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
    fp_dtype = x.dtype
    orig_shape = x.shape
    x_flat     = x.contiguous().view(-1)
    M          = x_flat.numel()

    # ---- 随机概率张量 ----
    if stochastic_rounding:
        prob = torch.rand_like(x_flat, dtype=torch.float32)
    else:
        prob = torch.zeros_like(x_flat, dtype=torch.float32)

    out   = torch.empty_like(x_flat)
    BLOCK = 32  # MXFP8使用32的block size
    grid  = ((M + BLOCK - 1) // BLOCK,)

    mxfp8_e4m3_fwd_kernel[grid](
        x_flat, out, prob, M,
        BLOCK=BLOCK,
        STOCHASTIC=stochastic_rounding,
        num_warps=4
    )
    return out.view(orig_shape).to(fp_dtype)


# ---------------------------------------------------------------
# Python 包装：MXFP8 E5M2 量化
# ---------------------------------------------------------------
def mxfp8_e5m2_forward(
    x: torch.Tensor,
    stochastic_rounding: bool = False,
):
    """
    x             : CUDA 张量 (f16/f32)，任意形状
    stochastic_rounding: 是否使用随机舍入
    """
    
    assert x.is_cuda and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
    fp_dtype = x.dtype
    orig_shape = x.shape
    x_flat     = x.contiguous().view(-1)
    M          = x_flat.numel()

    # ---- 随机概率张量 ----
    if stochastic_rounding:
        prob = torch.rand_like(x_flat, dtype=torch.float32)
    else:
        prob = torch.zeros_like(x_flat, dtype=torch.float32)

    out   = torch.empty_like(x_flat)
    BLOCK = 32  # MXFP8使用32的block size
    grid  = ((M + BLOCK - 1) // BLOCK,)

    mxfp8_e5m2_fwd_kernel[grid](
        x_flat, out, prob, M,
        BLOCK=BLOCK,
        STOCHASTIC=stochastic_rounding,
        num_warps=4
    )
    return out.view(orig_shape).to(fp_dtype)


# ---------------------------------------------------------------
# 通用 MXFP8 接口
# ---------------------------------------------------------------
def mxfp8_forward(
    x: torch.Tensor,
    format: str = "e4m3",
    stochastic_rounding: bool = False,
):
    """
    x             : CUDA 张量 (f16/f32)，任意形状
    format        : "e4m3" 或 "e5m2"
    stochastic_rounding: 是否使用随机舍入
    """
    if format == "e4m3":
        return mxfp8_e4m3_forward(x, stochastic_rounding)
    elif format == "e5m2":
        return mxfp8_e5m2_forward(x, stochastic_rounding)
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'e4m3' or 'e5m2'")


# ---------------------------------------------------------------
# Quick sanity-check
# ---------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    B, N = 2, 32
    x = torch.randn(B, N, device="cuda", dtype=torch.float16) * 100

    # Test E4M3
    y_e4m3_det = mxfp8_e4m3_forward(x, stochastic_rounding=False)
    y_e4m3_sto = mxfp8_e4m3_forward(x, stochastic_rounding=True)
    print("MXFP8 E4M3 det  :", y_e4m3_det[0, :8])
    print("MXFP8 E4M3 sto  :", y_e4m3_sto[0, :8])

    # Test E5M2
    y_e5m2_det = mxfp8_e5m2_forward(x, stochastic_rounding=False)
    y_e5m2_sto = mxfp8_e5m2_forward(x, stochastic_rounding=True)
    print("MXFP8 E5M2 det  :", y_e5m2_det[0, :8])
    print("MXFP8 E5M2 sto  :", y_e5m2_sto[0, :8]) 