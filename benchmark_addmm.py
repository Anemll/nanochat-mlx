"""
Micro-benchmark: Test if mx.addmm is slower than PyTorch's addmm
"""

import time
import numpy as np

MATRIX_SIZE = (1280, 1280)
NUM_WARMUP = 30
NUM_ITERATIONS = 1000  # More iterations for small operations

print("=" * 80)
print("Addmm Micro-Benchmark: PyTorch vs MLX")
print("=" * 80)
print(f"Matrix size: {MATRIX_SIZE}")
print(f"Warmup: {NUM_WARMUP}, Iterations: {NUM_ITERATIONS}")
print("=" * 80)

# ============================================================================
# PyTorch addmm
# ============================================================================
try:
    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"\n✓ PyTorch available (device: {device})")

    # Create matrices
    A = torch.randn(MATRIX_SIZE, device=device, dtype=torch.bfloat16)
    B = torch.randn(MATRIX_SIZE, device=device, dtype=torch.bfloat16)
    C = torch.randn(MATRIX_SIZE, device=device, dtype=torch.bfloat16)
    alpha = 2.0
    beta = 3.0

    # Warmup
    for _ in range(NUM_WARMUP):
        result = torch.addmm(C, A, B, beta=beta, alpha=alpha)
        if device == "mps":
            torch.mps.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(NUM_ITERATIONS):
        result = torch.addmm(C, A, B, beta=beta, alpha=alpha)
        if device == "mps":
            torch.mps.synchronize()
    end = time.perf_counter()

    pytorch_time = (end - start) / NUM_ITERATIONS * 1000
    print(f"PyTorch addmm: {pytorch_time:.3f} ms/iteration")
    PYTORCH_AVAILABLE = True

except Exception as e:
    print(f"PyTorch error: {e}")
    pytorch_time = None
    PYTORCH_AVAILABLE = False

# ============================================================================
# MLX addmm
# ============================================================================
try:
    import mlx.core as mx
    print(f"✓ MLX available")

    # Create matrices
    A_mlx = mx.random.normal(MATRIX_SIZE).astype(mx.bfloat16)
    B_mlx = mx.random.normal(MATRIX_SIZE).astype(mx.bfloat16)
    C_mlx = mx.random.normal(MATRIX_SIZE).astype(mx.bfloat16)
    alpha = 2.0
    beta = 3.0

    # Warmup
    for _ in range(NUM_WARMUP):
        result = mx.addmm(C_mlx, A_mlx, B_mlx, beta=beta, alpha=alpha)
        mx.eval(result)

    # Benchmark
    start = time.perf_counter()
    for _ in range(NUM_ITERATIONS):
        result = mx.addmm(C_mlx, A_mlx, B_mlx, beta=beta, alpha=alpha)
        mx.eval(result)
    end = time.perf_counter()

    mlx_time = (end - start) / NUM_ITERATIONS * 1000
    print(f"MLX addmm:     {mlx_time:.3f} ms/iteration")
    MLX_AVAILABLE = True

except Exception as e:
    print(f"MLX error: {e}")
    mlx_time = None
    MLX_AVAILABLE = False

# ============================================================================
# Also test plain matmul
# ============================================================================
print("\n" + "=" * 80)
print("Plain Matmul Comparison:")
print("=" * 80)

if PYTORCH_AVAILABLE:
    # Warmup
    for _ in range(NUM_WARMUP):
        result = A @ B
        if device == "mps":
            torch.mps.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(NUM_ITERATIONS):
        result = A @ B
        if device == "mps":
            torch.mps.synchronize()
    end = time.perf_counter()

    pytorch_matmul_time = (end - start) / NUM_ITERATIONS * 1000
    print(f"PyTorch matmul: {pytorch_matmul_time:.3f} ms/iteration")

if MLX_AVAILABLE:
    # Warmup
    for _ in range(NUM_WARMUP):
        result = A_mlx @ B_mlx
        mx.eval(result)

    # Benchmark
    start = time.perf_counter()
    for _ in range(NUM_ITERATIONS):
        result = A_mlx @ B_mlx
        mx.eval(result)
    end = time.perf_counter()

    mlx_matmul_time = (end - start) / NUM_ITERATIONS * 1000
    print(f"MLX matmul:     {mlx_matmul_time:.3f} ms/iteration")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 80)
print("Summary:")
print("=" * 80)

if pytorch_time and mlx_time:
    addmm_ratio = mlx_time / pytorch_time
    print(f"\naddmm ratio (MLX/PyTorch): {addmm_ratio:.2f}x")
    if addmm_ratio > 1:
        print(f"→ PyTorch addmm is {addmm_ratio:.2f}x FASTER")
    else:
        print(f"→ MLX addmm is {1/addmm_ratio:.2f}x FASTER")

if PYTORCH_AVAILABLE and MLX_AVAILABLE:
    matmul_ratio = mlx_matmul_time / pytorch_matmul_time
    print(f"\nmatmul ratio (MLX/PyTorch): {matmul_ratio:.2f}x")
    if matmul_ratio > 1:
        print(f"→ PyTorch matmul is {matmul_ratio:.2f}x FASTER")
    else:
        print(f"→ MLX matmul is {1/matmul_ratio:.2f}x FASTER")

print("\n" + "=" * 80)
print("\nNote: addmm computes: beta * C + alpha * (A @ B)")
print("This tests if the fusion itself is slower in MLX vs PyTorch MPS")
print("=" * 80)
