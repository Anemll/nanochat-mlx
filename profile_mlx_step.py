"""
Quick profiling script to identify MLX training bottlenecks.
"""
import time
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm.models.nanochat import Model as MLXNanoChatModel, ModelArgs as MLXModelArgs
from mlx_optimizers import Muon
from nanochat.dataset import parquets_iter_batched

# Quick model setup
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

# Get one batch
train_loader = parquets_iter_batched(split="train", start=0, step=1)
batch_loader = iter(train_loader)
tokens_batch = next(batch_loader)
x = mx.array(tokens_batch[:, :-1])
y = mx.array(tokens_batch[:, 1:])

# Define simple loss
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

print("=" * 60)
print("MLX TRAINING STEP PROFILING")
print("=" * 60)

# Time forward pass only
print("\n1. Forward pass only:")
for i in range(3):
    t0 = time.time()
    logits = model(x)
    mx.eval(logits)
    t1 = time.time()
    print(f"   Run {i+1}: {(t1-t0)*1000:.2f}ms")

# Time loss computation
print("\n2. Forward + Loss:")
for i in range(3):
    t0 = time.time()
    loss = compute_loss(model, x, y)
    mx.eval(loss)
    t1 = time.time()
    print(f"   Run {i+1}: {(t1-t0)*1000:.2f}ms | loss: {float(loss):.4f}")

# Time loss + gradients
print("\n3. Loss + Gradients:")
loss_and_grad_fn = lambda m, i, t: mx.value_and_grad(compute_loss, argnums=0)(m, i, t)
for i in range(3):
    t0 = time.time()
    loss, grads = loss_and_grad_fn(model, x, y)
    mx.eval(loss)
    mx.eval(grads)
    t1 = time.time()
    print(f"   Run {i+1}: {(t1-t0)*1000:.2f}ms | loss: {float(loss):.4f}")

# Time optimizer update (single optimizer, all params)
print("\n4. Loss + Gradients + Single Optimizer Update:")
single_optimizer = optim.AdamW(learning_rate=0.01)
for i in range(3):
    t0 = time.time()
    loss, grads = loss_and_grad_fn(model, x, y)
    single_optimizer.update(model, grads)
    mx.eval(model.parameters(), single_optimizer.state)
    t1 = time.time()
    print(f"   Run {i+1}: {(t1-t0)*1000:.2f}ms | loss: {float(loss):.4f}")

# Time multi-optimizer update (the complex path)
print("\n5. Loss + Gradients + Multi-Optimizer Update (current implementation):")
from mlx.utils import tree_map, tree_flatten

# Setup param groups (simplified)
def get_param_names(tree, prefix=""):
    names = []
    if isinstance(tree, dict):
        for k, v in tree.items():
            new_prefix = f"{prefix}.{k}" if prefix else k
            if isinstance(v, (dict, list)):
                names.extend(get_param_names(v, new_prefix))
            else:
                names.append(new_prefix)
    elif isinstance(tree, list):
        for idx, v in enumerate(tree):
            new_prefix = f"{prefix}.{idx}" if prefix else str(idx)
            if isinstance(v, (dict, list)):
                names.extend(get_param_names(v, new_prefix))
            else:
                names.append(new_prefix)
    return names

all_param_names = get_param_names(model.parameters())
embedding_names = [n for n in all_param_names if "wte" in n]
lm_head_names = [n for n in all_param_names if "lm_head" in n]
matrix_names = [n for n in all_param_names if n not in embedding_names + lm_head_names]

adamw_emb = optim.AdamW(learning_rate=0.01)
adamw_lm = optim.AdamW(learning_rate=0.01)
muon_opt = Muon(learning_rate=0.02, momentum=0.95, nesterov=True, backend='newtonschulz5', backend_steps=5)

param_groups = [
    {"name": "embedding", "optimizer": adamw_emb, "param_names": embedding_names},
    {"name": "lm_head", "optimizer": adamw_lm, "param_names": lm_head_names},
    {"name": "matrix", "optimizer": muon_opt, "param_names": matrix_names},
]

def extract_group_grads(all_grads, param_names, model_params):
    def navigate_and_extract(grad_tree, param_tree, current_path=""):
        if isinstance(grad_tree, dict) and isinstance(param_tree, dict):
            result = {}
            for key in grad_tree.keys():
                if key in param_tree:
                    new_path = f"{current_path}.{key}" if current_path else key
                    extracted = navigate_and_extract(grad_tree[key], param_tree[key], new_path)
                    if extracted is not None:
                        result[key] = extracted
            return result if result else None
        elif isinstance(grad_tree, list) and isinstance(param_tree, list):
            result = []
            for idx, (grad_item, param_item) in enumerate(zip(grad_tree, param_tree)):
                new_path = f"{current_path}.{idx}" if current_path else str(idx)
                extracted = navigate_and_extract(grad_item, param_item, new_path)
                result.append(extracted)
            return result if any(x is not None for x in result) else None
        else:
            if current_path in param_names:
                return grad_tree
            return None
    return navigate_and_extract(all_grads, model_params)

for i in range(3):
    t0 = time.time()
    loss, grads = loss_and_grad_fn(model, x, y)

    t_grad_split_start = time.time()
    optimizer_states = []
    for group in param_groups:
        group_grads = extract_group_grads(grads, group["param_names"], model.parameters())
        if group_grads is not None:
            group["optimizer"].update(model, group_grads)
            optimizer_states.append(group["optimizer"].state)
    t_grad_split_end = time.time()

    mx.eval(model.parameters(), *optimizer_states)
    t1 = time.time()

    grad_split_time = (t_grad_split_end - t_grad_split_start) * 1000
    total_time = (t1 - t0) * 1000
    print(f"   Run {i+1}: {total_time:.2f}ms total | grad split: {grad_split_time:.2f}ms ({grad_split_time/total_time*100:.1f}%)")

print("\n" + "=" * 60)
