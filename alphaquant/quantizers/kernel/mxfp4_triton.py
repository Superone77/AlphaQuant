from random import randint

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 32 * 32}),
        triton.Config({"BLOCK_SIZE": 64 * 32}),
        triton.Config({"BLOCK_SIZE": 128 * 32}),
        triton.Config({"BLOCK_SIZE": 256 * 32}),
        triton.Config({"BLOCK_SIZE": 512 * 32}),
    ],
    key=[],
)
@triton.jit
def mxfp4_forward_kernel(
    x_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    group_size: tl.constexpr,
    stochastic_round: tl.constexpr,
    seed: int,
    BLOCK_SIZE: tl.constexpr,
):    
    
    # load x
    pid = tl.program_id(0)
    start_idx = pid * BLOCK_SIZE
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x_flat = tl.load(x_ptr + offsets, mask=mask)
    
    # hadamard transform
    
    # group
    x_had_grouped = tl.reshape(x_had, (BLOCK_SIZE // group_size, group_size))
    
    # scale
    scales = tl.max(tl.abs(x_had_grouped), axis=-1, keep_dims=True)
    shared_exps = tl.exp2(tl.floor(tl.log2(scales)) - 2) / (3/4)
    x_had_scaled = x_had_grouped / shared_exps
    
    # quantize
    x_had_scaled_abs = tl.abs(x_had_scaled)
    x_had_scaled_sign = tl.where(
        x_had_scaled > 0,
        1,
        -1,
    )
    if stochastic_round:
        x_fp4_high = tl.where(
            x_had_scaled_abs > 4,
            6,
            tl.where(
                x_had_scaled_abs > 3,
                4,
                tl.where(
                    x_had_scaled_abs > 2,
                    3,
                    tl.where(
                        x_had_scaled_abs > 1.5,
                        2,
                        tl.where(
                            x_had_scaled_abs > 1.0,
                            1.5,
                            tl.where(
                                x_had_scaled_abs > 0.5,
                                1,
                                0.5,
                            )
                        )
                    )
                )
            )
        )
        
        x_fp4_low = tl.where(
            x_had_scaled_abs > 4,
            4,
            tl.where(
                x_had_scaled_abs > 3,
                3,
                tl.where(
                    x_had_scaled_abs > 2,
                    2,
                    tl.where(
                        x_had_scaled_abs > 1.5,
                        1.5,
                        tl.where(
                            x_had_scaled_abs > 1.0,
                            1.0,
                            tl.where(
                                x_had_scaled_abs > 0.5,
                                0.5,
                                0.0,
                            )
                        )
                    )
                )
            )
        )
        
        prob_up = (x_had_scaled_abs - x_fp4_low) / (x_fp4_high - x_fp4_low)
        sampled_prob = tl.rand(seed, offsets)
        x_fp4 = tl.where(
            sampled_prob < prob_up,
            x_fp4_high,
            x_fp4_low,
        ) * x_had_scaled_sign
    else:
        x_fp4 = tl.where(
            x_had_scaled_abs > 5,
            6,
            tl.where(
                x_had_scaled_abs > 3.5,
                4,
                tl.where(
                    x_had_scaled_abs > 2.5,
                    3,
                    tl.where(
                        x_had_scaled_abs > 1.75,
                        2,
                        tl.where(
                            x_had_scaled_abs > 1.25,
                            1.5,
                            tl.where(
                                x_had_scaled_abs > 0.75,
                                1,
                                tl.where(
                                    x_had_scaled_abs > 0.25,
                                    0.5,
                                    0,
                                )
                            )
                        )
                    )
                )
            )
        ) * x_had_scaled_sign

    # dequantize
    x_dequantized = x_fp4 * shared_exps
    
    # Reshape back to flat form for storage
    x_dequantized_flat = tl.reshape(x_dequantized, (BLOCK_SIZE,))
    
    # store
    tl.store(output_ptr + offsets, x_dequantized_flat, mask=mask)


@torch.compiler.disable()
def mxfp4_forward_kernel_wrapper(
    x,
    stochastic_round=False,
):    
    # Make sure inputs are contiguous
    x = x.contiguous()
    
    # Create output tensor
    output = torch.empty_like(x)
    
    
    if stochastic_round:
        seed = randint(0, 1000000)
    else:
        seed = None
    
    # Get total number of elements and calculate grid for launching the kernel
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    
    # Launch optimized kernel
    mxfp4_forward_kernel[grid](
        x_ptr=x,
        output_ptr=output,
        n_elements=n_elements,
        group_size=32,
        stochastic_round=stochastic_round,
        seed=seed,
    )
    
    return output
