"""
Test MLX compilation effectiveness for Newton-Schulz.
Compares uncompiled vs compiled vs full-graph compiled versions.
"""

import time
import mlx.core as mx

# Test configuration
SHAPE = (1280, 1280)  # Typical transformer weight
NUM_WARMUP = 10
NUM_ITERATIONS = 100
NS_STEPS = 5

print("=" * 80)
print("MLX Compilation Test for Newton-Schulz")
print("=" * 80)
print(f"Matrix shape: {SHAPE}")
print(f"Newton-Schulz steps: {NS_STEPS}")
print(f"Warmup: {NUM_WARMUP}, Iterations: {NUM_ITERATIONS}")
print("=" * 80)

# ============================================================================
# Version 1: Uncompiled (baseline)
# ============================================================================
def newton_schulz_uncompiled(G, steps=5):
    """Newton-Schulz without compilation"""
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

# ============================================================================
# Version 2: Compiled function
# ============================================================================
@mx.compile
def newton_schulz_compiled(G, steps=5):
    """Newton-Schulz with @mx.compile decorator"""
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

# ============================================================================
# Version 3: Manually call mx.compile()
# ============================================================================
def newton_schulz_manual(G, steps=5):
    """Newton-Schulz compiled manually with mx.compile()"""
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

# Manually compile it
newton_schulz_manual_compiled = mx.compile(newton_schulz_manual)

# ============================================================================
# Version 4: MLX Native Muon (from mlx.optimizers)
# ============================================================================
from mlx.optimizers import Muon

# Extract the method
muon_instance = Muon(learning_rate=0.02, ns_steps=5)

def newton_schulz_native(G, steps=5):
    """Newton-Schulz from mlx.optimizers.Muon"""
    return muon_instance._zeropower_via_newtonschulz5(G, steps)

# ============================================================================
# Benchmark function
# ============================================================================
def benchmark(func, name, shape=SHAPE, steps=NS_STEPS):
    """Benchmark a Newton-Schulz implementation"""
    print(f"\n{'─' * 80}")
    print(f"Testing: {name}")
    print(f"{'─' * 80}")

    # Create test matrix
    G = mx.random.normal(shape).astype(mx.bfloat16)

    # Warmup
    print(f"  Warming up ({NUM_WARMUP} iterations)...")
    for i in range(NUM_WARMUP):
        result = func(G, steps=steps)
        mx.eval(result)

    # Benchmark
    print(f"  Benchmarking ({NUM_ITERATIONS} iterations)...")
    start = time.perf_counter()
    for _ in range(NUM_ITERATIONS):
        result = func(G, steps=steps)
        mx.eval(result)
    end = time.perf_counter()

    avg_time_ms = (end - start) / NUM_ITERATIONS * 1000
    print(f"  ✓ Average time: {avg_time_ms:.3f} ms/iteration")

    return avg_time_ms

# ============================================================================
# Run all benchmarks
# ============================================================================
print("\n" + "=" * 80)
print("Running Benchmarks...")
print("=" * 80)

results = []

# Test 1: Uncompiled
t1 = benchmark(newton_schulz_uncompiled, "1. Uncompiled (baseline)")
results.append(("Uncompiled", t1))

# Test 2: @mx.compile decorator
t2 = benchmark(newton_schulz_compiled, "2. @mx.compile decorator")
results.append(("@mx.compile", t2))

# Test 3: Manual mx.compile()
t3 = benchmark(newton_schulz_manual_compiled, "3. Manual mx.compile()")
results.append(("Manual compile", t3))

# Test 4: MLX Native
t4 = benchmark(newton_schulz_native, "4. MLX Native (mlx.optimizers.Muon)")
results.append(("MLX Native", t4))

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 80)
print("Summary:")
print("=" * 80)
print(f"\n{'Implementation':<30} {'Time (ms)':<12} {'vs Uncompiled':<15} {'vs Native':<12}")
print("─" * 80)

baseline = results[0][1]
native = results[3][1]

for name, time_ms in results:
    speedup_vs_baseline = baseline / time_ms
    speedup_vs_native = time_ms / native
    print(f"{name:<30} {time_ms:>9.3f}    {speedup_vs_baseline:>10.2f}x      {speedup_vs_native:>8.2f}x")

print("\n" + "=" * 80)
print("Key Findings:")
print("=" * 80)

# Check if compilation helps
if results[1][1] < results[0][1] * 0.95:
    speedup = results[0][1] / results[1][1]
    print(f"✓ Compilation HELPS: {speedup:.2f}x speedup vs uncompiled")
else:
    print(f"✗ Compilation DOESN'T HELP: only {results[0][1]/results[1][1]:.2f}x")

# Check if native is fastest
if results[3][1] < min([r[1] for r in results[:3]]):
    print(f"✓ MLX Native is FASTEST")
else:
    fastest = min(results[:3], key=lambda x: x[1])
    print(f"✗ {fastest[0]} is faster than MLX Native by {results[3][1]/fastest[1]:.2f}x")

# Check consistency between compiled versions
if abs(results[1][1] - results[2][1]) / results[1][1] < 0.05:
    print(f"✓ @mx.compile and manual compile are equivalent (within 5%)")
else:
    print(f"✗ @mx.compile and manual compile differ by {abs(results[1][1]-results[2][1])/results[1][1]*100:.1f}%")

print("\n" + "=" * 80)
