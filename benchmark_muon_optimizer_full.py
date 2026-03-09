"""
Full Muon Optimizer Benchmark: PyTorch vs MLX
Tests the complete optimizer update step including momentum and parameter updates.
"""

import time
import numpy as np

# Test configuration
MATRIX_SIZES = [
    (1280, 1280),  # Typical square attention/MLP weight
    (1280, 320),   # Typical rectangular weight
    (5120, 1280),  # Typical MLP weight (4x expansion)
]
NUM_WARMUP = 30
NUM_ITERATIONS = 100
NS_STEPS = 5

print("=" * 80)
print("Full Muon Optimizer Benchmark: PyTorch vs MLX")
print("=" * 80)
print(f"\nNewton-Schulz steps: {NS_STEPS}")
print(f"Warmup iterations: {NUM_WARMUP}")
print(f"Benchmark iterations: {NUM_ITERATIONS}")
print("\n" + "=" * 80)

# ============================================================================
# PyTorch Muon Optimizer
# ============================================================================
try:
    import torch
    from torch.optim import Optimizer as TorchOptimizer

    PYTORCH_AVAILABLE = True
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"\n✓ PyTorch available (device: {device})")

    @torch.compile
    def zeropower_via_newtonschulz5(G, steps=5):
        """PyTorch Newton-Schulz implementation"""
        a, b, c = (3.4445, -4.7750, 2.0315)
        X = G.bfloat16()
        X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)

        if G.size(-2) > G.size(-1):
            X = X.mT

        for _ in range(steps):
            A = X @ X.mT
            B = b * A + c * A @ A
            X = a * X + B @ X

        if G.size(-2) > G.size(-1):
            X = X.mT
        return X

    class PyTorchMuon(TorchOptimizer):
        """PyTorch Muon optimizer"""
        def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
            defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
            super().__init__(params, defaults)

        @torch.no_grad()
        def step(self):
            for group in self.param_groups:
                lr = group['lr']
                momentum = group['momentum']
                nesterov = group['nesterov']
                ns_steps = group['ns_steps']

                for p in group['params']:
                    if p.grad is None:
                        continue

                    grad = p.grad
                    state = self.state[p]

                    # Initialize momentum buffer
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(p)

                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(grad)

                    if nesterov:
                        grad_update = grad + momentum * buf
                    else:
                        grad_update = buf

                    # Apply Newton-Schulz for 2D parameters
                    if grad_update.ndim >= 2:
                        grad_update = zeropower_via_newtonschulz5(grad_update, steps=ns_steps)
                        # Aspect ratio scaling
                        grad_update = grad_update * (max(1, grad_update.shape[-2] / grad_update.shape[-1]) ** 0.5)

                    p.add_(grad_update, alpha=-lr)

    def benchmark_pytorch(shape):
        """Benchmark PyTorch Muon optimizer"""
        # Create parameter and gradient
        param = torch.randn(shape, device=device, dtype=torch.bfloat16, requires_grad=True)
        grad = torch.randn(shape, device=device, dtype=torch.bfloat16)

        # Create optimizer
        optimizer = PyTorchMuon([param], lr=0.02, momentum=0.95, nesterov=True, ns_steps=NS_STEPS)

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

        avg_time_ms = (end - start) / NUM_ITERATIONS * 1000
        return avg_time_ms

except ImportError as e:
    PYTORCH_AVAILABLE = False
    print(f"\n✗ PyTorch not available: {e}")
    def benchmark_pytorch(shape):
        return None

# ============================================================================
# MLX Muon Optimizer
# ============================================================================
try:
    import mlx.core as mx
    from mlx.optimizers import Muon as MLXMuon

    MLX_AVAILABLE = True
    print(f"✓ MLX available")

    def benchmark_mlx(shape):
        """Benchmark MLX Muon optimizer"""
        import mlx.nn as nn

        # Create a simple model with one parameter
        class SimpleModel(nn.Module):
            def __init__(self, shape):
                super().__init__()
                self.weight = mx.random.normal(shape).astype(mx.bfloat16)

        model = SimpleModel(shape)

        # Create gradient
        grad = mx.random.normal(shape).astype(mx.bfloat16)
        grads = {"weight": grad}

        # Create MLX Muon optimizer
        optimizer = MLXMuon(learning_rate=0.02, momentum=0.95, nesterov=True, ns_steps=NS_STEPS)

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
print("Results (Full Optimizer Update):")
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
            print(f"  PyTorch Muon: {pytorch_time:.3f} ms/iteration")
        except Exception as e:
            print(f"  PyTorch Muon: ERROR - {e}")

    if MLX_AVAILABLE:
        try:
            mlx_time = benchmark_mlx(shape)
            print(f"  MLX Muon:     {mlx_time:.3f} ms/iteration")
        except Exception as e:
            print(f"  MLX Muon:     ERROR - {e}")

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
print("\nNote: This benchmark tests the full optimizer.update() including:")
print("  - Momentum buffer updates")
print("  - Newton-Schulz orthogonalization (5 steps)")
print("  - Aspect ratio scaling")
print("  - Parameter updates")
print("=" * 80)
