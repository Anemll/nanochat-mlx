"""
Isolated benchmark: PyTorch Muon vs MLX Muon
Tests Newton-Schulz performance on identical matrices with 5 steps.
"""

import time
import numpy as np

# Test configurations
MATRIX_SIZES = [
    (1280, 1280),  # Typical square attention/MLP weight
    (1280, 320),   # Typical rectangular weight
    (5120, 1280),  # Typical MLP weight (4x expansion)
]
NUM_WARMUP = 30  # Increased to ensure torch.compile is fully warmed up
NUM_ITERATIONS = 100
NS_STEPS = 5

print("=" * 80)
print("Muon Newton-Schulz Benchmark: PyTorch MPS vs MLX")
print("=" * 80)
print(f"\nNewton-Schulz steps: {NS_STEPS}")
print(f"Warmup iterations: {NUM_WARMUP}")
print(f"Benchmark iterations: {NUM_ITERATIONS}")
print("\n" + "=" * 80)

# ============================================================================
# PyTorch Implementation
# ============================================================================
try:
    import torch
    from nanochat.muon import zeropower_via_newtonschulz5

    PYTORCH_AVAILABLE = True
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"\n✓ PyTorch available (device: {device})")

    def benchmark_pytorch(shape):
        """Benchmark PyTorch Newton-Schulz"""
        # Create test matrix
        G = torch.randn(shape, device=device, dtype=torch.bfloat16)

        # Warmup
        for _ in range(NUM_WARMUP):
            result = zeropower_via_newtonschulz5(G, steps=NS_STEPS)
            if device == "mps":
                torch.mps.synchronize()

        # Benchmark
        start = time.perf_counter()
        for _ in range(NUM_ITERATIONS):
            result = zeropower_via_newtonschulz5(G, steps=NS_STEPS)
            if device == "mps":
                torch.mps.synchronize()
        end = time.perf_counter()

        avg_time_ms = (end - start) / NUM_ITERATIONS * 1000
        return avg_time_ms

except ImportError as e:
    PYTORCH_AVAILABLE = False
    print(f"\n✗ PyTorch not available: {e}")
    def benchmark_pytorch(shape):
        return None

# ============================================================================
# MLX Implementation
# ============================================================================
try:
    import mlx.core as mx
    from mlx.optimizers import Muon

    MLX_AVAILABLE = True
    print(f"✓ MLX available")

    # Extract the Newton-Schulz function from MLX's Muon
    def mlx_newton_schulz(G, steps=5):
        """MLX Newton-Schulz implementation (from mlx.optimizers.Muon)"""
        assert G.ndim == 2, f"Expected 2D array, got shape {G.shape}"

        a, b, c = (3.4445, -4.7750, 2.0315)
        transpose_needed = G.shape[-2] > G.shape[-1]

        if transpose_needed:
            X = G.T
        else:
            X = G

        X = X / (mx.linalg.norm(X, keepdims=True) + 1e-7)

        for _ in range(steps):
            A = X @ X.T
            B = mx.addmm(b * A, A, A, beta=1.0, alpha=c)
            X = mx.addmm(a * X, B, X, beta=1.0, alpha=1.0)

        if transpose_needed:
            X = X.T

        return X

    def benchmark_mlx(shape):
        """Benchmark MLX Newton-Schulz"""
        # Create test matrix
        G = mx.random.normal(shape).astype(mx.bfloat16)

        # Warmup
        for _ in range(NUM_WARMUP):
            result = mlx_newton_schulz(G, steps=NS_STEPS)
            mx.eval(result)

        # Benchmark
        start = time.perf_counter()
        for _ in range(NUM_ITERATIONS):
            result = mlx_newton_schulz(G, steps=NS_STEPS)
            mx.eval(result)
        end = time.perf_counter()

        avg_time_ms = (end - start) / NUM_ITERATIONS * 1000
        return avg_time_ms

except ImportError as e:
    MLX_AVAILABLE = False
    print(f"✗ MLX not available: {e}")
    def benchmark_mlx(shape):
        return None

# ============================================================================
# Run Benchmarks
# ============================================================================
print("\n" + "=" * 80)
print("Results:")
print("=" * 80)

results = []

for shape in MATRIX_SIZES:
    print(f"\n{'─' * 80}")
    print(f"Matrix shape: {shape[0]} × {shape[1]}")
    print(f"{'─' * 80}")

    pytorch_time = None
    mlx_time = None

    if PYTORCH_AVAILABLE:
        try:
            pytorch_time = benchmark_pytorch(shape)
            print(f"  PyTorch MPS: {pytorch_time:.3f} ms/iteration")
        except Exception as e:
            print(f"  PyTorch MPS: ERROR - {e}")

    if MLX_AVAILABLE:
        try:
            mlx_time = benchmark_mlx(shape)
            print(f"  MLX:         {mlx_time:.3f} ms/iteration")
        except Exception as e:
            print(f"  MLX:         ERROR - {e}")

    if pytorch_time and mlx_time:
        speedup = mlx_time / pytorch_time
        print(f"  Ratio (MLX/PyTorch): {speedup:.2f}x")
        if speedup > 1:
            print(f"  → PyTorch is {speedup:.2f}x FASTER")
        else:
            print(f"  → MLX is {1/speedup:.2f}x FASTER")
        results.append((shape, pytorch_time, mlx_time, speedup))

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 80)
print("Summary:")
print("=" * 80)

if results:
    print(f"\n{'Shape':<20} {'PyTorch (ms)':<15} {'MLX (ms)':<15} {'MLX/PyTorch':<12}")
    print("─" * 80)
    for shape, pt_time, mlx_time, ratio in results:
        shape_str = f"{shape[0]}×{shape[1]}"
        print(f"{shape_str:<20} {pt_time:>12.3f}    {mlx_time:>12.3f}    {ratio:>9.2f}x")

    # Overall statistics
    avg_ratio = np.mean([r[3] for r in results])
    print("\n" + "─" * 80)
    print(f"Average MLX/PyTorch ratio: {avg_ratio:.2f}x")
    if avg_ratio > 1:
        print(f"→ PyTorch is {avg_ratio:.2f}x faster on average")
    else:
        print(f"→ MLX is {1/avg_ratio:.2f}x faster on average")
else:
    print("\nNo results to compare (need both PyTorch and MLX)")

print("\n" + "=" * 80)
