"""Check if bfloat16 is actually being used in forward/backward."""
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.nanochat import Model as MLXNanoChatModel, ModelArgs as MLXModelArgs
from mlx.utils import tree_map as mlx_tree_map

# Create small model
model_args = MLXModelArgs(
    model_type="nanochat",
    vocab_size=65536,
    hidden_size=256,
    num_hidden_layers=2,
    num_attention_heads=2,
    num_key_value_heads=2,
    intermediate_size=1024,
    max_position_embeddings=1024,
    rope_theta=10000.0,
)
model = MLXNanoChatModel(model_args)

# Convert to bfloat16
def to_bfloat16(x):
    if isinstance(x, mx.array) and x.dtype == mx.float32:
        return x.astype(mx.bfloat16)
    return x

model.update(mlx_tree_map(to_bfloat16, model.parameters()))

print("=" * 60)
print("BFLOAT16 DTYPE VERIFICATION")
print("=" * 60)

# Check parameter dtypes
params = model.parameters()
print("\n1. Parameter dtypes:")
print(f"   wte.weight: {params['transformer']['wte']['weight'].dtype}")
print(f"   h[0].attn.c_q.weight: {params['transformer']['h'][0]['attn']['c_q']['weight'].dtype}")

# Create inputs
x = mx.array([[1, 2, 3, 4, 5]])
y = mx.array([[2, 3, 4, 5, 6]])

# Forward pass
print("\n2. Forward pass dtypes:")
logits = model(x)
print(f"   Input dtype: {x.dtype}")
print(f"   Logits dtype: {logits.dtype}")

# Check intermediate activations by inspecting first layer
h = model.transformer.wte(x)
print(f"   Embedding output dtype: {h.dtype}")

# Compute loss
def compute_loss(model, inputs, targets):
    logits = model(inputs)
    B, T, V = logits.shape
    logits_flat = logits.reshape(-1, V)
    targets_flat = targets.reshape(-1).astype(mx.int32)
    max_logits = mx.max(logits_flat, axis=-1, keepdims=True)
    exp_logits = mx.exp(logits_flat - max_logits)
    log_probs = (logits_flat - max_logits) - mx.log(mx.sum(exp_logits, axis=-1, keepdims=True))
    batch_indices = mx.arange(logits_flat.shape[0])
    target_log_probs = log_probs[batch_indices, targets_flat]
    loss = -mx.mean(target_log_probs)
    return loss

print("\n3. Loss computation dtypes:")
loss = compute_loss(model, x, y)
print(f"   Loss dtype: {loss.dtype}")

# Compute gradients
print("\n4. Gradient dtypes:")
loss_and_grad = nn.value_and_grad(model, compute_loss)
loss, grads = loss_and_grad(model, x, y)
print(f"   wte.weight grad dtype: {grads['transformer']['wte']['weight'].dtype}")
print(f"   h[0].attn.c_q.weight grad dtype: {grads['transformer']['h'][0]['attn']['c_q']['weight'].dtype}")

# Check if gradients match parameter dtype
grad_dtype = grads['transformer']['wte']['weight'].dtype
param_dtype = params['transformer']['wte']['weight'].dtype
print(f"\n5. Gradient/Parameter dtype match: {grad_dtype == param_dtype}")

if grad_dtype == mx.float32:
    print("\n⚠️  WARNING: Gradients are FLOAT32, not BFLOAT16!")
    print("   This means backward pass is using FP32, defeating the purpose!")
elif grad_dtype == mx.bfloat16:
    print("\n✅ SUCCESS: Gradients are BFLOAT16!")

print("\n" + "=" * 60)
