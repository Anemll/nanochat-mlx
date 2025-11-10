"""Test if MLX supports float16/bfloat16."""
import mlx.core as mx

print("Available dtypes in MLX:")
print(f"  float32: {mx.float32}")
print(f"  float16: {mx.float16}")

# Check if bfloat16 exists
try:
    print(f"  bfloat16: {mx.bfloat16}")
    has_bf16 = True
except AttributeError:
    print("  bfloat16: NOT AVAILABLE")
    has_bf16 = False

# Test float16
x = mx.array([1.0, 2.0, 3.0], dtype=mx.float16)
print(f"\nFloat16 array dtype: {x.dtype}")
print(f"Float16 works: {x.dtype == mx.float16}")

if has_bf16:
    x_bf16 = mx.array([1.0, 2.0, 3.0], dtype=mx.bfloat16)
    print(f"Bfloat16 array dtype: {x_bf16.dtype}")
    print(f"Bfloat16 works: {x_bf16.dtype == mx.bfloat16}")
