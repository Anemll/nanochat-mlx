"""Check optimizer state memory usage."""
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm.models.nanochat import Model as MLXNanoChatModel, ModelArgs as MLXModelArgs
from mlx.utils import tree_map as mlx_tree_map
from mlx_optimizers import Muon

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
print("OPTIMIZER STATE DTYPE CHECK")
print("=" * 60)

# Create optimizers
adamw_opt = optim.AdamW(learning_rate=0.01, betas=(0.8, 0.95), eps=1e-10)
muon_opt = Muon(learning_rate=0.02, momentum=0.95, nesterov=True, backend='newtonschulz5', backend_steps=5)

# Create dummy gradients (bfloat16)
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

print(f"\n1. Model parameter dtype: {model.parameters()['transformer']['wte']['weight'].dtype}")
print(f"2. Gradient dtype: {grads['transformer']['wte']['weight'].dtype}")

# Apply AdamW update
adamw_opt.update(model, grads)
mx.eval(model.parameters(), adamw_opt.state)

# Check optimizer state dtypes
print(f"\n3. AdamW optimizer state:")
if adamw_opt.state:
    # Get first parameter's state
    wte_state = adamw_opt.state[0]['transformer']['wte']['weight']
    print(f"   State keys: {wte_state.keys()}")
    print(f"   'm' (momentum) dtype: {wte_state['m'].dtype}")
    print(f"   'v' (variance) dtype: {wte_state['v'].dtype}")
    
    if wte_state['m'].dtype == mx.float32:
        print("\n   ⚠️  AdamW stores optimizer state in FLOAT32!")
        print("      This is EXPECTED and correct for mixed precision training")
        print("      BUT it doubles memory usage compared to full BF16")

# Check Muon
muon_opt.update(model, grads)
mx.eval(model.parameters(), muon_opt.state)

print(f"\n4. Muon optimizer state:")
if muon_opt.state:
    wte_muon_state = muon_opt.state[0]['transformer']['wte']['weight']
    print(f"   State keys: {wte_muon_state.keys()}")
    if 'momentum' in wte_muon_state:
        print(f"   'momentum' dtype: {wte_muon_state['momentum'].dtype}")

print("\n" + "=" * 60)
print("MEMORY ANALYSIS:")
print("=" * 60)
print("\nWith BF16 model + FP32 optimizer states:")
print("  - Parameters: BF16 (2 bytes/param)")
print("  - Gradients: BF16 (2 bytes/param) - temporary")
print("  - AdamW state: FP32 (4 bytes × 2 = 8 bytes/param for m+v)")
print("  - Total per param: 2 + 8 = 10 bytes")
print("\nWith FP32 everything:")
print("  - Parameters: FP32 (4 bytes/param)")
print("  - Gradients: FP32 (4 bytes/param) - temporary")
print("  - AdamW state: FP32 (8 bytes/param for m+v)")  
print("  - Total per param: 4 + 8 = 12 bytes")
print("\nExpected memory reduction: 10/12 = 83% of FP32")
print("(Not 50% because optimizer states dominate!)")
print("=" * 60)
