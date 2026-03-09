"""
Test CoreML matmul performance vs MLX vs PyTorch
This helps determine if the matmul slowdown is MLX-specific or Metal-wide
"""

import time
import numpy as np

# Rectangular matrix multiplication: (M×K) @ (K×N) → (M×N)
M = 1280
K = 1280
N = 128
MATRIX_A_SIZE = (M, K)
MATRIX_B_SIZE = (K, N)
NUM_WARMUP = 30
NUM_ITERATIONS = 1000

print("=" * 80)
print("CoreML vs MLX vs PyTorch Matmul Benchmark")
print("=" * 80)
print(f"Matrix A: {MATRIX_A_SIZE}, Matrix B: {MATRIX_B_SIZE}")
print(f"Result: ({M}×{N})")
print(f"Warmup: {NUM_WARMUP}, Iterations: {NUM_ITERATIONS}")
print("=" * 80)

# ============================================================================
# Create and export CoreML model
# ============================================================================
try:
    import coremltools as ct

    print("\n✓ CoreMLTools available")
    print("Creating CoreML matmul model...")

    # Use traced PyTorch model for CoreML conversion (simpler and more reliable)
    import torch

    class MatmulModel(torch.nn.Module):
        def forward(self, a, b):
            return torch.matmul(a, b)

    model = MatmulModel()
    model.eval()

    # Create example inputs
    example_a = torch.randn(MATRIX_A_SIZE, dtype=torch.float16)
    example_b = torch.randn(MATRIX_B_SIZE, dtype=torch.float16)

    # Trace the model
    traced_model = torch.jit.trace(model, (example_a, example_b))

    # Convert to CoreML
    coreml_model = ct.convert(
        traced_model,
        inputs=[
            ct.TensorType(name="a", shape=MATRIX_A_SIZE, dtype=np.float16),
            ct.TensorType(name="b", shape=MATRIX_B_SIZE, dtype=np.float16),
         ],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS18,
        compute_units=ct.ComputeUnit.CPU_AND_GPU # Use Neural Engine + GPU
    )

    # Save model
    model_path = "/tmp/matmul_test.mlpackage"
    coreml_model.save(model_path)
    print(f"✓ CoreML model saved to {model_path}")

    # Note: CoreML models are automatically compiled on first prediction
    # The .mlpackage format is optimized and compiled by the runtime

    # Prepare inputs (float16 for CoreML)
    A_np = np.random.randn(*MATRIX_A_SIZE).astype(np.float16)
    B_np = np.random.randn(*MATRIX_B_SIZE).astype(np.float16)

    # Warmup
    for _ in range(NUM_WARMUP):
        result = coreml_model.predict({'a': A_np, 'b': B_np})

    # Benchmark
    start = time.perf_counter()
    for _ in range(NUM_ITERATIONS):
        result = coreml_model.predict({'a': A_np, 'b': B_np})
    end = time.perf_counter()

    coreml_time = (end - start) / NUM_ITERATIONS * 1000
    print(f"CoreML matmul: {coreml_time:.3f} ms/iteration")
    COREML_AVAILABLE = True

except ImportError:
    print("\n✗ CoreMLTools not available (install: pip install coremltools)")
    coreml_time = None
    COREML_AVAILABLE = False
except Exception as e:
    print(f"\n✗ CoreML error: {e}")
    coreml_time = None
    COREML_AVAILABLE = False

# ============================================================================
# PyTorch MPS matmul
# ============================================================================
try:
    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"\n✓ PyTorch available (device: {device})")

    A_torch = torch.randn(MATRIX_A_SIZE, device=device, dtype=torch.float16)
    B_torch = torch.randn(MATRIX_B_SIZE, device=device, dtype=torch.float16)

    # Warmup
    for _ in range(NUM_WARMUP):
        result = A_torch @ B_torch
        if device == "mps":
            torch.mps.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(NUM_ITERATIONS):
        result = A_torch @ B_torch
        if device == "mps":
            torch.mps.synchronize()
    end = time.perf_counter()

    pytorch_time = (end - start) / NUM_ITERATIONS * 1000
    print(f"PyTorch matmul (float16): {pytorch_time:.3f} ms/iteration")
    PYTORCH_AVAILABLE = True

except Exception as e:
    print(f"\n✗ PyTorch error: {e}")
    pytorch_time = None
    PYTORCH_AVAILABLE = False

# ============================================================================
# MLX matmul
# ============================================================================
try:
    import mlx.core as mx
    print(f"✓ MLX available")

    A_mlx = mx.random.normal(MATRIX_A_SIZE).astype(mx.float16)
    B_mlx = mx.random.normal(MATRIX_B_SIZE).astype(mx.float16)

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

    mlx_time = (end - start) / NUM_ITERATIONS * 1000
    print(f"MLX matmul (float16):     {mlx_time:.3f} ms/iteration")
    MLX_AVAILABLE = True

except Exception as e:
    print(f"\n✗ MLX error: {e}")
    mlx_time = None
    MLX_AVAILABLE = False

# ============================================================================
# Also test float16 (for comparison with Muon benchmark)
# ============================================================================
print("\n" + "=" * 80)
print("FP16 Comparison (PyTorch vs MLX):")
print("=" * 80)

if PYTORCH_AVAILABLE:
    A_torch_fp16 = torch.randn(MATRIX_A_SIZE, device=device, dtype=torch.float16)
    B_torch_fp16 = torch.randn(MATRIX_B_SIZE, device=device, dtype=torch.float16)

    # Warmup
    for _ in range(NUM_WARMUP):
        result = A_torch_fp16 @ B_torch_fp16
        if device == "mps":
            torch.mps.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(NUM_ITERATIONS):
        result = A_torch_fp16 @ B_torch_fp16
        if device == "mps":
            torch.mps.synchronize()
    end = time.perf_counter()

    pytorch_fp16_time = (end - start) / NUM_ITERATIONS * 1000
    print(f"PyTorch matmul (float16): {pytorch_fp16_time:.3f} ms/iteration")

if MLX_AVAILABLE:
    A_mlx_fp16 = mx.random.normal(MATRIX_A_SIZE).astype(mx.float16)
    B_mlx_fp16 = mx.random.normal(MATRIX_B_SIZE).astype(mx.float16)

    # Warmup
    for _ in range(NUM_WARMUP):
        result = A_mlx_fp16 @ B_mlx_fp16
        mx.eval(result)

    # Benchmark
    start = time.perf_counter()
    for _ in range(NUM_ITERATIONS):
        result = A_mlx_fp16 @ B_mlx_fp16
        mx.eval(result)
    end = time.perf_counter()

    mlx_fp16_time = (end - start) / NUM_ITERATIONS * 1000
    print(f"MLX matmul (float16):     {mlx_fp16_time:.3f} ms/iteration")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 80)
print("Summary (float16):")
print("=" * 80)

# results = []
# if COREML_AVAILABLE and coreml_time:
#     results.append(("CoreML", coreml_time))
# if PYTORCH_AVAILABLE and pytorch_time:
#     results.append(("PyTorch", pytorch_time))
# if MLX_AVAILABLE and mlx_time:
#     results.append(("MLX", mlx_time))

# if results:
#     print(f"\n{'Framework':<15} {'Time (ms)':<12} {'Ratio vs Fastest':<15}")
#     print("─" * 80)

#     fastest = min(results, key=lambda x: x[1])
#     fastest_time = fastest[1]

#     for name, time_ms in results:
#         ratio = time_ms / fastest_time
#         print(f"{name:<15} {time_ms:>9.3f}    {ratio:>9.2f}x")

if PYTORCH_AVAILABLE and MLX_AVAILABLE:
    print("\n" + "─" * 80)
    print(f"FP16 comparison:")
    fp16_ratio = mlx_fp16_time / pytorch_fp16_time
    print(f"  PyTorch fp16: {pytorch_fp16_time:.3f} ms")
    print(f"  MLX fp16:     {mlx_fp16_time:.3f} ms")
    print(f"  Ratio: {fp16_ratio:.2f}x (PyTorch is {fp16_ratio:.2f}x faster)" if fp16_ratio > 1 else f"  Ratio: {1/fp16_ratio:.2f}x (MLX is {1/fp16_ratio:.2f}x faster)")

print("\n" + "=" * 80)
print("\nConclusions:")
print("  - If CoreML ≈ PyTorch: MLX has matmul optimization opportunity")
print("  - If CoreML ≈ MLX: PyTorch MPS has superior matmul implementation")
print("  - This helps identify whether it's MLX-specific or Metal-wide issue")
print("=" * 80)
