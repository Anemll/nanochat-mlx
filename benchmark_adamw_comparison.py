"""
Test if matmul slowdown affects AdamW too
AdamW doesn't use matmul - it only does element-wise operations
"""

import time
import numpy as np

MATRIX_SIZE = (1280, 1280)
NUM_WARMUP = 30
NUM_ITERATIONS = 100

print("=" * 80)
print("AdamW Optimizer Benchmark: PyTorch vs MLX")
print("=" * 80)
print(f"Matrix size: {MATRIX_SIZE}")
print(f"Warmup: {NUM_WARMUP}, Iterations: {NUM_ITERATIONS}")
print("=" * 80)

# ============================================================================
# PyTorch AdamW
# ============================================================================
try:
    import torch
    from torch.optim import AdamW as TorchAdamW

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"\n✓ PyTorch available (device: {device})")

    # Create parameter
    param = torch.randn(MATRIX_SIZE, device=device, dtype=torch.bfloat16, requires_grad=True)
    grad = torch.randn(MATRIX_SIZE, device=device, dtype=torch.bfloat16)

    # Create optimizer
    optimizer = TorchAdamW([param], lr=0.001)

    # Warmup
    for _ in range(NUM_WARMUP):
        param.grad = grad
        optimizer.step()
        if device == "mps":
            torch.mps.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(NUM_ITERATIONS):
        param.grad = grad
        optimizer.step()
        if device == "mps":
            torch.mps.synchronize()
    end = time.perf_counter()

    pytorch_time = (end - start) / NUM_ITERATIONS * 1000
    print(f"PyTorch AdamW: {pytorch_time:.3f} ms/iteration")
    PYTORCH_AVAILABLE = True

except Exception as e:
    print(f"PyTorch error: {e}")
    pytorch_time = None
    PYTORCH_AVAILABLE = False

# ============================================================================
# MLX AdamW
# ============================================================================
try:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.optimizers import AdamW as MLXAdamW

    print(f"✓ MLX available")

    # Create simple model
    class SimpleModel(nn.Module):
        def __init__(self, shape):
            super().__init__()
            self.weight = mx.random.normal(shape).astype(mx.bfloat16)

    model = SimpleModel(MATRIX_SIZE)
    grad = mx.random.normal(MATRIX_SIZE).astype(mx.bfloat16)
    grads = {"weight": grad}

    # Create optimizer
    optimizer = MLXAdamW(learning_rate=0.001)

    # Warmup
    for _ in range(NUM_WARMUP):
        optimizer.update(model, grads)
        mx.eval(model.parameters())

    # Benchmark
    start = time.perf_counter()
    for _ in range(NUM_ITERATIONS):
        optimizer.update(model, grads)
        mx.eval(model.parameters())
    end = time.perf_counter()

    mlx_time = (end - start) / NUM_ITERATIONS * 1000
    print(f"MLX AdamW:     {mlx_time:.3f} ms/iteration")
    MLX_AVAILABLE = True

except Exception as e:
    print(f"MLX error: {e}")
    mlx_time = None
    MLX_AVAILABLE = False

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 80)
print("Summary:")
print("=" * 80)

if pytorch_time and mlx_time:
    ratio = mlx_time / pytorch_time
    print(f"\nAdamW ratio (MLX/PyTorch): {ratio:.2f}x")
    if ratio > 1:
        print(f"→ PyTorch is {ratio:.2f}x FASTER")
    else:
        print(f"→ MLX is {1/ratio:.2f}x FASTER")

print("\n" + "=" * 80)
print("\nNote: AdamW uses only element-wise operations (no matmul)")
print("If AdamW shows similar performance, matmul is NOT the issue")
print("If AdamW is much closer in speed, matmul IS the bottleneck")
print("=" * 80)
