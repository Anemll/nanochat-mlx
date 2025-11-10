"""Test converting MLX model to bfloat16."""
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.nanochat import Model as MLXNanoChatModel, ModelArgs as MLXModelArgs

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

print("Before conversion:")
params = model.parameters()
print(f"  wte.weight dtype: {params['transformer']['wte']['weight'].dtype}")
print(f"  h[0].attn.c_q.weight dtype: {params['transformer']['h'][0]['attn']['c_q']['weight'].dtype}")

# Try to convert to bfloat16 using mx.utils.tree_map
from mlx.utils import tree_map

def to_bfloat16(x):
    if isinstance(x, mx.array) and x.dtype == mx.float32:
        return x.astype(mx.bfloat16)
    return x

# Convert all parameters
model.update(tree_map(to_bfloat16, model.parameters()))

print("\nAfter conversion:")
params = model.parameters()
print(f"  wte.weight dtype: {params['transformer']['wte']['weight'].dtype}")
print(f"  h[0].attn.c_q.weight dtype: {params['transformer']['h'][0]['attn']['c_q']['weight'].dtype}")

# Test forward pass
x = mx.array([[1, 2, 3, 4, 5]])
logits = model(x)
print(f"\nLogits dtype after forward: {logits.dtype}")
print("Forward pass works!")

# Test gradients
y = mx.array([[2, 3, 4, 5, 6]])

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

loss_and_grad = nn.value_and_grad(model, compute_loss)
loss, grads = loss_and_grad(model, x, y)

print(f"\nLoss dtype: {loss.dtype}")
print(f"Gradient dtype: {grads['transformer']['wte']['weight'].dtype}")
print("Gradient computation works!")

print("\n✅ BFloat16 training is supported in MLX!")
