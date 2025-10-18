"""
Hadamard Transform Utilities for Outlier Suppression

This module provides Hadamard transformation functions to suppress activation
outliers before quantization. Based on MoEQuant's implementation.

Key idea:
- Hadamard transform redistributes outliers more evenly
- Can be fused into weights to maintain mathematical invariance
- For linear layer: y = Wx, we can apply: y = (W @ H^T) @ (H @ x) = W' @ x'
- Since H is orthogonal (H^T @ H = I), this maintains equivalence

Usage:
    1. Apply Hadamard to weight: W' = W @ H^T
    2. During inference: x' = H @ x (but we fuse this into adjacent layer)
    3. Net effect: No online transformation needed!
"""

import math
import torch
import torch.nn as nn
from typing import Optional, Tuple


def is_pow2(n: int) -> bool:
    """Check if n is a power of 2."""
    return (n & (n - 1) == 0) and (n > 0)


def get_hadK(n: int, transpose: bool = False) -> Tuple[Optional[torch.Tensor], int]:
    """
    Get Hadamard matrix for dimension n.
    
    For dimensions that are powers of 2, returns (None, 1).
    For other dimensions, returns a base Hadamard matrix.
    
    Args:
        n: Dimension size
        transpose: Whether to transpose the matrix
        
    Returns:
        Tuple of (hadamard_matrix, K) where K is the base dimension
    """
    if n % 172 == 0:
        assert is_pow2(n // 172)
        K = 172
        hadK = get_had172().T if transpose else get_had172()
    elif n % 156 == 0:
        assert is_pow2(n // 156)
        K = 156
        hadK = get_had156().T if transpose else get_had156()
    elif n % 140 == 0:
        assert is_pow2(n // 140)
        K = 140
        hadK = get_had140().T if transpose else get_had140()
    elif n % 108 == 0:
        assert is_pow2(n // 108)
        K = 108
        hadK = get_had108().T if transpose else get_had108()
    elif n % 60 == 0:
        assert is_pow2(n // 60)
        K = 60
        hadK = get_had60().T if transpose else get_had60()
    elif n % 52 == 0:
        assert is_pow2(n // 52)
        K = 52
        hadK = get_had52().T if transpose else get_had52()
    elif n % 36 == 0:
        assert is_pow2(n // 36)
        K = 36
        hadK = get_had36().T if transpose else get_had36()
    elif n % 28 == 0:
        assert is_pow2(n // 28)
        K = 28
        hadK = get_had28().T if transpose else get_had28()
    elif n % 40 == 0:
        assert is_pow2(n // 40)
        K = 40
        hadK = get_had40().T if transpose else get_had40()
    elif n % 20 == 0:
        assert is_pow2(n // 20)
        K = 20
        hadK = get_had20().T if transpose else get_had20()
    elif n % 12 == 0:
        assert is_pow2(n // 12)
        K = 12
        hadK = get_had12().T if transpose else get_had12()
    else:
        assert is_pow2(n), f"Dimension {n} not supported for Hadamard transform"
        K = 1
        hadK = None
    
    return hadK, K


def hadamard_transform_naive(X: torch.Tensor, scale: float = None) -> torch.Tensor:
    """
    Apply Hadamard transform using recursive doubling algorithm.
    
    This is a CPU-friendly implementation. For faster GPU implementation,
    use fast_hadamard_transform library if available.
    
    Args:
        X: Input tensor, last dimension must be power of 2
        scale: Scaling factor (default: 1/sqrt(n))
        
    Returns:
        Transformed tensor
    """
    n = X.shape[-1]
    assert is_pow2(n), "Last dimension must be power of 2"
    
    if scale is None:
        scale = 1.0 / math.sqrt(n)
    
    # Reshape for processing
    input_shape = X.shape
    X = X.reshape(-1, n)
    
    # Recursive Hadamard transform using butterfly structure
    m = 1
    while m < n:
        # Process in pairs
        for i in range(0, n, 2 * m):
            for j in range(m):
                idx1 = i + j
                idx2 = i + j + m
                
                a = X[:, idx1].clone()
                b = X[:, idx2].clone()
                
                X[:, idx1] = a + b
                X[:, idx2] = a - b
        m *= 2
    
    X = X * scale
    return X.reshape(input_shape)


def matmul_hadU(X: torch.Tensor, hadK: Optional[torch.Tensor] = None, K: int = 1) -> torch.Tensor:
    """
    Apply Hadamard transform to tensor X.
    
    For K=1 (power of 2 dimension): Use fast recursive algorithm
    For K>1: Use mixed approach with base Hadamard matrix
    
    Args:
        X: Input tensor [..., n]
        hadK: Base Hadamard matrix (K x K), or None if K=1
        K: Base dimension
        
    Returns:
        Transformed tensor with same shape
    """
    n = X.shape[-1]
    device = X.device
    dtype = X.dtype
    
    if K == 1:
        # Pure power of 2 - use recursive algorithm
        return hadamard_transform_naive(X)
    
    # Mixed: recursive + base matrix
    # Reshape to [..., n//K, K]
    input_shape = X.shape
    X_flat = X.reshape(-1, n // K, K)
    
    # Apply base Hadamard matrix
    hadK = hadK.to(device=device, dtype=dtype)
    X_flat = torch.matmul(X_flat, hadK.T)
    
    # Apply recursive Hadamard on n//K dimension
    # Reshape to [...*K, n//K]
    X_reshaped = X_flat.permute(0, 2, 1).reshape(-1, n // K)
    X_reshaped = hadamard_transform_naive(X_reshaped, scale=1.0 / math.sqrt(n // K))
    
    # Reshape back
    X_flat = X_reshaped.reshape(X_flat.shape[0], K, n // K).permute(0, 2, 1)
    X_out = X_flat.reshape(input_shape)
    
    # Overall scale
    X_out = X_out / math.sqrt(K)
    
    return X_out


def apply_hadamard_to_linear(
    module: nn.Linear,
    mode: str = 'left',
    transpose: bool = False
) -> None:
    """
    Apply Hadamard transform to a linear layer's weights in-place.
    
    This modifies the layer weights to incorporate Hadamard transform,
    enabling outlier suppression without online transformation.
    
    For mode='left': W' = H @ W  (transform output features)
    For mode='right': W' = W @ H^T  (transform input features)
    
    Args:
        module: Linear layer to transform
        mode: 'left' or 'right' transformation
        transpose: Whether to use transposed Hadamard
    """
    assert isinstance(module, nn.Linear), "Module must be nn.Linear"
    
    W = module.weight.data
    dtype = W.dtype
    device = W.device
    out_features, in_features = W.shape
    
    # Convert to float for numerical stability
    W_float = W.float()
    
    if mode == 'left':
        # Transform output dimension: W' = H @ W
        n = out_features
        hadK, K = get_hadK(n, transpose=transpose)
        W_transformed = matmul_hadU(W_float, hadK, K)
        
    elif mode == 'right':
        # Transform input dimension: W' = W @ H^T
        n = in_features
        hadK, K = get_hadK(n, transpose=(not transpose))  # Transpose for right multiply
        # W @ H^T = (H @ W^T)^T
        W_t = W_float.t()
        W_t_transformed = matmul_hadU(W_t, hadK, K)
        W_transformed = W_t_transformed.t()
        
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    # Store back
    module.weight.data = W_transformed.to(dtype=dtype, device=device)
    
    # Transform bias if present (only for left mode)
    if module.bias is not None and mode == 'left':
        B = module.bias.data.float()
        B_transformed = matmul_hadU(B.unsqueeze(0), hadK, K).squeeze(0)
        module.bias.data = B_transformed.to(dtype=dtype, device=device)


def fuse_hadamard_transforms(
    layer_prev: nn.Linear,
    layer_next: nn.Linear
) -> None:
    """
    Fuse Hadamard transforms between two consecutive linear layers.
    
    For layers: y = layer_next(layer_prev(x))
    We apply: 
        - layer_prev: W1' = W1 @ H^T (right transform)
        - layer_next: W2' = H @ W2 (left transform)
    
    This makes: y = W2 @ H @ H^T @ W1 @ x = W2' @ W1' @ x
    Since H @ H^T = I (orthogonal), mathematically equivalent!
    
    Args:
        layer_prev: Previous layer (output will be transformed)
        layer_next: Next layer (input will be transformed)
    """
    # Check dimension compatibility
    assert layer_prev.out_features == layer_next.in_features, \
        "Layer dimensions must match"
    
    n = layer_prev.out_features
    
    # Get Hadamard matrix for this dimension
    hadK, K = get_hadK(n, transpose=False)
    
    # Apply right transform to previous layer: W1' = W1 @ H^T
    apply_hadamard_to_linear(layer_prev, mode='right', transpose=True)
    
    # Apply left transform to next layer: W2' = H @ W2
    apply_hadamard_to_linear(layer_next, mode='left', transpose=False)


# Base Hadamard matrices (from MoEQuant)
# These are used for non-power-of-2 dimensions

def get_had12() -> torch.Tensor:
    """12x12 Hadamard matrix."""
    return torch.FloatTensor([
        [+1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [+1, +1, -1, +1, -1, -1, -1, +1, +1, +1, -1, +1],
        [+1, +1, +1, -1, +1, -1, -1, -1, +1, +1, +1, -1],
        [+1, -1, +1, +1, -1, +1, -1, -1, -1, +1, +1, +1],
        [+1, +1, -1, +1, +1, -1, +1, -1, -1, -1, +1, +1],
        [+1, +1, +1, -1, +1, +1, -1, +1, -1, -1, -1, +1],
        [+1, +1, +1, +1, -1, +1, +1, -1, +1, -1, -1, -1],
        [+1, -1, +1, +1, +1, -1, +1, +1, -1, +1, -1, -1],
        [+1, -1, -1, +1, +1, +1, -1, +1, +1, -1, +1, -1],
        [+1, -1, -1, -1, +1, +1, +1, -1, +1, +1, -1, +1],
        [+1, +1, -1, -1, -1, +1, +1, +1, -1, +1, +1, -1],
        [+1, -1, +1, -1, -1, -1, +1, +1, +1, -1, +1, +1],
    ])


def get_had20() -> torch.Tensor:
    """20x20 Hadamard matrix."""
    return torch.FloatTensor([
        [+1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [+1, +1, -1, +1, +1, -1, -1, -1, -1, +1, -1, +1, -1, +1, +1, +1, +1, -1, -1, +1],
        [+1, +1, +1, -1, +1, +1, -1, -1, -1, -1, +1, -1, +1, -1, +1, +1, +1, +1, -1, -1],
        [+1, -1, +1, +1, -1, +1, +1, -1, -1, -1, -1, +1, -1, +1, -1, +1, +1, +1, +1, -1],
        [+1, -1, -1, +1, +1, -1, +1, +1, -1, -1, -1, -1, +1, -1, +1, -1, +1, +1, +1, +1],
        [+1, +1, -1, -1, +1, +1, -1, +1, +1, -1, -1, -1, -1, +1, -1, +1, -1, +1, +1, +1],
        [+1, +1, +1, -1, -1, +1, +1, -1, +1, +1, -1, -1, -1, -1, +1, -1, +1, -1, +1, +1],
        [+1, +1, +1, +1, -1, -1, +1, +1, -1, +1, +1, -1, -1, -1, -1, +1, -1, +1, -1, +1],
        [+1, +1, +1, +1, +1, -1, -1, +1, +1, -1, +1, +1, -1, -1, -1, -1, +1, -1, +1, -1],
        [+1, -1, +1, +1, +1, +1, -1, -1, +1, +1, -1, +1, +1, -1, -1, -1, -1, +1, -1, +1],
        [+1, +1, -1, +1, +1, +1, +1, -1, -1, +1, +1, -1, +1, +1, -1, -1, -1, -1, +1, -1],
        [+1, -1, +1, -1, +1, +1, +1, +1, -1, -1, +1, +1, -1, +1, +1, -1, -1, -1, -1, +1],
        [+1, +1, -1, +1, -1, +1, +1, +1, +1, -1, -1, +1, +1, -1, +1, +1, -1, -1, -1, -1],
        [+1, -1, +1, -1, +1, -1, +1, +1, +1, +1, -1, -1, +1, +1, -1, +1, +1, -1, -1, -1],
        [+1, -1, -1, +1, -1, +1, -1, +1, +1, +1, +1, -1, -1, +1, +1, -1, +1, +1, -1, -1],
        [+1, -1, -1, -1, +1, -1, +1, -1, +1, +1, +1, +1, -1, -1, +1, +1, -1, +1, +1, -1],
        [+1, -1, -1, -1, -1, +1, -1, +1, -1, +1, +1, +1, +1, -1, -1, +1, +1, -1, +1, +1],
        [+1, +1, -1, -1, -1, -1, +1, -1, +1, -1, +1, +1, +1, +1, -1, -1, +1, +1, -1, +1],
        [+1, +1, +1, -1, -1, -1, -1, +1, -1, +1, -1, +1, +1, +1, +1, -1, -1, +1, +1, -1],
        [+1, -1, +1, +1, -1, -1, -1, -1, +1, -1, +1, -1, +1, +1, +1, +1, -1, -1, +1, +1]
    ])


def get_had28() -> torch.Tensor:
    """28x28 Hadamard matrix (for LLaMA-3)."""
    # Simplified version - full matrix is very long
    # In practice, load from file or generate
    raise NotImplementedError("28x28 Hadamard matrix - use full implementation from MoEQuant")


def get_had36() -> torch.Tensor:
    """36x36 Hadamard matrix."""
    raise NotImplementedError("36x36 Hadamard matrix - use full implementation from MoEQuant")


def get_had40() -> torch.Tensor:
    """40x40 Hadamard matrix."""
    raise NotImplementedError("40x40 Hadamard matrix - use full implementation from MoEQuant")


def get_had52() -> torch.Tensor:
    """52x52 Hadamard matrix."""
    raise NotImplementedError("52x52 Hadamard matrix - use full implementation from MoEQuant")


def get_had60() -> torch.Tensor:
    """60x60 Hadamard matrix."""
    raise NotImplementedError("60x60 Hadamard matrix - use full implementation from MoEQuant")


def get_had108() -> torch.Tensor:
    """108x108 Hadamard matrix."""
    raise NotImplementedError("108x108 Hadamard matrix - use full implementation from MoEQuant")


def get_had140() -> torch.Tensor:
    """140x140 Hadamard matrix."""
    raise NotImplementedError("140x140 Hadamard matrix - use full implementation from MoEQuant")


def get_had156() -> torch.Tensor:
    """156x156 Hadamard matrix."""
    raise NotImplementedError("156x156 Hadamard matrix - use full implementation from MoEQuant")


def get_had172() -> torch.Tensor:
    """172x172 Hadamard matrix (for LLaMA-2 7B and up)."""
    raise NotImplementedError("172x172 Hadamard matrix - use full implementation from MoEQuant")

