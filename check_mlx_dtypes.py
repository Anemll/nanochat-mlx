"""Check data types used in MLX model and gradients."""
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.nanochat import Model as MLXNanoChatModel, ModelArgs as MLXModelArgs

# Create small model
model_args = MLXModelArgs(
    model_type="nanochat",
    vocab_size=65536,
    hidden_size=256,
    num_hidden_layers=4,
    num_attention_heads=2,
    num_key_value_heads=2,
    intermediate_size=1024,
    max_position_embeddings=1024,
    rope_theta=10000.0,
)
model = MLXNanoChatModel(model_args)

print("=" * 60)
print("MLX DATA TYPES CHECK")
print("=" * 60)

# Check parameter dtypes
print("\nParameter dtypes:")
params = model.parameters()
print(f"  wte.weight: {params['transformer']['wte']['weight'].dtype}")
print(f"  lm_head.weight: {params['lm_head']['weight'].dtype}")
print(f"  h[0].attn.c_q.weight: {params['transformer']['h'][0]['attn']['c_q']['weight'].dtype}")
print(f"  h[0].mlp.c_fc.weight: {params['transformer']['h'][0]['mlp']['c_fc']['weight'].dtype}")

# Check what dtype MLX uses by default
print(f"\nMLX default float dtype: {mx.float32}")
test_array = mx.array([1.0, 2.0, 3.0])
print(f"Default array dtype: {test_array.dtype}")

# Create dummy input and compute gradients
x = mx.array([[1, 2, 3, 4, 5]])
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
print(f"\nGradient dtypes:")
print(f"  wte.weight grad: {grads['transformer']['wte']['weight'].dtype}")
print(f"  lm_head.weight grad: {grads['lm_head']['weight'].dtype}")
print(f"  h[0].attn.c_q.weight grad: {grads['transformer']['h'][0]['attn']['c_q']['weight'].dtype}")

# Check intermediate computation dtypes
logits = model(x)
print(f"\nIntermediate dtypes:")
print(f"  Logits: {logits.dtype}")

print("\n" + "=" * 60)
