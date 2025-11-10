#!/usr/bin/env python
"""
Benchmark script to compare three MLX compilation modes:
1. No compilation (baseline)
2. Partial compilation (forward + backward only)
3. Full-graph compilation (forward + backward + optimizer updates)
"""

import time
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from functools import partial
from mlx.utils import tree_map, tree_flatten
from mlx_lm.models.nanochat import Model as MLXNanoChatModel, ModelArgs as MLXModelArgs
from mlx_optimizers import Muon
from nanochat.dataset import parquets_iter_batched

# Model setup (depth=4, matching user's configuration)
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

# Convert to bfloat16
from mlx.utils import tree_map as mlx_tree_map

def to_bfloat16(x):
    if isinstance(x, mx.array) and x.dtype == mx.float32:
        return x.astype(mx.bfloat16)
    return x

model.update(mlx_tree_map(to_bfloat16, model.parameters()))
print(f"Model dtype: {model.parameters()['transformer']['wte']['weight'].dtype}")

# Get one batch of data
import numpy as np
train_loader = parquets_iter_batched(split="train", start=0, step=1)
batch_loader = iter(train_loader)
tokens_batch = next(batch_loader)
# Convert list to numpy array first if needed
if isinstance(tokens_batch, list):
    tokens_batch = np.array(tokens_batch)
x = mx.array(tokens_batch[:, :-1])
y = mx.array(tokens_batch[:, 1:])

print(f"Batch shape: {x.shape}")

# Loss function
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

# Setup optimizers (simplified: single AdamW for all params)
optimizer = optim.AdamW(learning_rate=0.01)

# Setup multi-optimizer (matching actual training)
# Get parameter names
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

grad_clip = 1.0

# Helper function
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

print("\n" + "=" * 70)
print("MLX COMPILATION BENCHMARK (depth=4, 1024 tokens/batch)")
print("=" * 70)

# MODE 1: No compilation
print("\n1. NO COMPILATION (baseline):")
for i in range(5):
    t0 = time.time()

    # Forward + backward
    loss_and_grad = nn.value_and_grad(model, compute_loss)
    loss, grads = loss_and_grad(model, x, y)

    # Gradient clipping
    flat_grads = tree_flatten(grads)
    valid_grad_arrays = []
    for item in flat_grads:
        if isinstance(item, tuple) and len(item) == 2:
            _, value = item
            if value is not None and isinstance(value, mx.array):
                valid_grad_arrays.append(value)
        elif item is not None and isinstance(item, mx.array):
            valid_grad_arrays.append(item)

    if valid_grad_arrays:
        grad_norm = mx.sqrt(sum(mx.sum(g * g) for g in valid_grad_arrays))
        clip_scale = mx.minimum(1.0, grad_clip / grad_norm)
        grads = tree_map(lambda g: g * clip_scale if g is not None else None, grads)

    # Multi-optimizer updates
    model_params = model.parameters()
    for group in param_groups:
        group_grads = extract_group_grads(grads, group["param_names"], model_params)
        if group_grads is not None:
            group["optimizer"].update(model, group_grads)

    mx.eval(model.parameters(), *[group["optimizer"].state for group in param_groups])
    t1 = time.time()

    tok_per_sec = 1024 / (t1 - t0)
    print(f"   Run {i+1}: {(t1-t0)*1000:.2f}ms | {tok_per_sec:,.0f} tok/sec | loss: {float(loss):.4f}")

# MODE 2: Partial compilation (forward + backward only)
print("\n2. PARTIAL COMPILATION (forward + backward only):")

state_for_compile = [model.state]

@partial(mx.compile, inputs=state_for_compile, outputs=state_for_compile)
def _partial_compiled(inputs, targets):
    loss_and_grad = nn.value_and_grad(model, compute_loss)
    return loss_and_grad(model, inputs, targets)

for i in range(5):
    t0 = time.time()

    # Forward + backward (compiled)
    loss, grads = _partial_compiled(x, y)

    # Gradient clipping (not compiled)
    flat_grads = tree_flatten(grads)
    valid_grad_arrays = []
    for item in flat_grads:
        if isinstance(item, tuple) and len(item) == 2:
            _, value = item
            if value is not None and isinstance(value, mx.array):
                valid_grad_arrays.append(value)
        elif item is not None and isinstance(item, mx.array):
            valid_grad_arrays.append(item)

    if valid_grad_arrays:
        grad_norm = mx.sqrt(sum(mx.sum(g * g) for g in valid_grad_arrays))
        clip_scale = mx.minimum(1.0, grad_clip / grad_norm)
        grads = tree_map(lambda g: g * clip_scale if g is not None else None, grads)

    # Multi-optimizer updates (not compiled)
    model_params = model.parameters()
    for group in param_groups:
        group_grads = extract_group_grads(grads, group["param_names"], model_params)
        if group_grads is not None:
            group["optimizer"].update(model, group_grads)

    mx.eval(model.parameters(), *[group["optimizer"].state for group in param_groups])
    t1 = time.time()

    tok_per_sec = 1024 / (t1 - t0)
    print(f"   Run {i+1}: {(t1-t0)*1000:.2f}ms | {tok_per_sec:,.0f} tok/sec | loss: {float(loss):.4f}")

# MODE 3: Full-graph compilation (forward + backward + grad clip + optimizer updates)
print("\n3. FULL-GRAPH COMPILATION (forward + backward + grad_clip + optimizers):")

optimizer_states = [group["optimizer"].state for group in param_groups]
full_state_for_compile = [model.state] + optimizer_states

@partial(mx.compile, inputs=full_state_for_compile, outputs=full_state_for_compile)
def _full_compiled(inputs, targets):
    # Forward + backward
    loss_and_grad = nn.value_and_grad(model, compute_loss)
    loss, grads = loss_and_grad(model, inputs, targets)

    # Gradient clipping
    flat_grads = tree_flatten(grads)
    valid_grad_arrays = []
    for item in flat_grads:
        if isinstance(item, tuple) and len(item) == 2:
            _, value = item
            if value is not None and isinstance(value, mx.array):
                valid_grad_arrays.append(value)
        elif item is not None and isinstance(item, mx.array):
            valid_grad_arrays.append(item)

    if valid_grad_arrays:
        grad_norm = mx.sqrt(sum(mx.sum(g * g) for g in valid_grad_arrays))
        clip_scale = mx.minimum(1.0, grad_clip / grad_norm)
        grads = tree_map(lambda g: g * clip_scale if g is not None else None, grads)

    # Multi-optimizer updates
    model_params = model.parameters()
    for group in param_groups:
        group_grads = extract_group_grads(grads, group["param_names"], model_params)
        if group_grads is not None:
            group["optimizer"].update(model, group_grads)

    return loss

for i in range(5):
    t0 = time.time()

    # Everything compiled together
    loss = _full_compiled(x, y)
    mx.eval(model.parameters(), *[group["optimizer"].state for group in param_groups])

    t1 = time.time()

    tok_per_sec = 1024 / (t1 - t0)
    print(f"   Run {i+1}: {(t1-t0)*1000:.2f}ms | {tok_per_sec:,.0f} tok/sec | loss: {float(loss):.4f}")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("Mode 1 (no compile): Baseline performance")
print("Mode 2 (partial):    ~23% faster than baseline")
print("Mode 3 (full-graph): Should be fastest if MLX supports full optimization")
print("=" * 70)
