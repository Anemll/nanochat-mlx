# MLX Port Issues - Detailed Comparison

## Executive Summary

The MLX port ([base_train_mlx.py](scripts/base_train_mlx.py)) has **critical optimizer setup issues** that prevent the model from learning properly. Loss stays at ~17 (vs PyTorch ~7.5 after 100 steps), and sampling produces gibberish.

---

## Critical Issues

### 1. **Single Optimizer Instead of Separate AdamW + Muon** ⚠️ CRITICAL

**PyTorch ([base_train.py:177-178](scripts/base_train.py#L177-L178)):**
```python
optimizers = model.setup_optimizers(
    unembedding_lr=unembedding_lr,  # 0.004
    embedding_lr=embedding_lr,       # 0.2
    matrix_lr=matrix_lr,             # 0.02
    weight_decay=weight_decay
)
adamw_optimizer, muon_optimizer = optimizers  # Two separate optimizers!
```

**MLX ([base_train_mlx.py:508-513](scripts/base_train_mlx.py#L508-L513)):**
```python
# Create single AdamW optimizer for all parameters
base_lr = (embedding_lr + unembedding_lr) / 2 * dmodel_lr_scale
optimizer = optim.AdamW(
    learning_rate=base_lr,  # WRONG: Averaging LRs
    betas=(0.8, 0.95),
    eps=1e-10,
    weight_decay=weight_decay
)
# Only ONE optimizer! Missing Muon entirely!
```

**Impact:**
- Matrix parameters (Q/K/V/proj in attention, FC/proj in MLP) use AdamW instead of Muon
- Muon uses momentum-based updates optimized for matrix operations
- Wrong optimizer → poor gradient updates → model doesn't learn

---

### 2. **Incorrect Learning Rate Scaling** ⚠️ CRITICAL

**PyTorch ([base_train.py:321-324](scripts/base_train.py#L321-L324)):**
```python
lrm = get_lr_multiplier(step)
for opt in optimizers:
    for group in opt.param_groups:
        group["lr"] = group["initial_lr"] * lrm  # Scale from initial_lr
```

The `setup_optimizers()` method creates param groups with different `initial_lr`:
- Embedding: `embedding_lr` (0.2)
- LM head: `unembedding_lr` (0.004)
- Matrix (Muon): `matrix_lr` (0.02)

**MLX ([base_train_mlx.py:684-686](scripts/base_train_mlx.py#L684-L686)):**
```python
lrm = get_lr_multiplier(step)
# Update optimizer learning rate
optimizer.learning_rate = base_lr * lrm  # WRONG: All params get same LR
```

**Impact:**
- All parameters get same LR = `(0.2 + 0.004)/2 * dmodel_lr_scale = 0.102 * scale`
- Embedding should get 0.2, lm_head should get 0.004
- Wrong LR scaling → inefficient training

---

### 3. **Missing Muon Momentum Scheduling** ⚠️ HIGH

**PyTorch ([base_train.py:206-210, 325-327](scripts/base_train.py#L206-L210)):**
```python
# Momentum scheduler for Muon optimizer
def get_muon_momentum(it):
    frac = min(it / 300, 1)
    momentum = (1 - frac) * 0.85 + frac * 0.95
    return momentum

# In training loop:
muon_momentum = get_muon_momentum(step)
for group in muon_optimizer.param_groups:
    group["momentum"] = muon_momentum
```

**MLX:** Completely missing! No Muon optimizer, no momentum scheduling.

**Impact:** Matrix parameters don't get the adaptive momentum schedule that Muon provides.

---

### 4. **Validation Not Implemented** ⚠️ HIGH

**PyTorch ([base_train.py:224-240](scripts/base_train.py#L224-L240)):**
```python
if last_step or step % eval_every == 0:
    model.eval()
    val_loader = build_val_loader()
    eval_steps = eval_tokens // (device_batch_size * max_seq_len * ddp_world_size)
    with autocast_ctx:
        val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
    print0(f"Step {step:05d} | Validation bpb: {val_bpb:.4f}")
```

**MLX ([base_train_mlx.py:551-563](scripts/base_train_mlx.py#L551-L563)):**
```python
if last_step or step % eval_every == 0:
    # TODO: Implement evaluate_bpb for MLX
    val_bpb = 0.0  # Placeholder
    print0(f"Step {step:05d} | Validation bpb: {val_bpb:.4f}")
```

**Impact:** Can't measure validation performance, can't detect overfitting.

---

### 5. **CORE Metric Evaluation Disabled** ⚠️ MEDIUM

**PyTorch ([base_train.py:242-256](scripts/base_train.py#L242-L256)):**
```python
if core_metric_every > 0 and (last_step or (step > 0 and step % core_metric_every == 0)):
    model.eval()
    with autocast_ctx:
        results = evaluate_model(orig_model, tokenizer, device, max_per_task=core_metric_max_per_task)
    print0(f"Step {step:05d} | CORE metric: {results['core_metric']:.4f}")
```

**MLX ([base_train_mlx.py:177](scripts/base_train_mlx.py#L177)):**
```python
core_metric_every = -1  # disable CORE metric for now
```

**Impact:** Can't measure model performance on downstream tasks.

---

### 6. **No Checkpoint Saving** ⚠️ MEDIUM

**PyTorch ([base_train.py:280-298](scripts/base_train.py#L280-L298)):**
```python
if master_process and last_step:
    output_dirname = checkpoint_model_tag if checkpoint_model_tag else f"d{checkpoint_depth}"
    checkpoint_dir = os.path.join(base_dir, "base_checkpoints", output_dirname)
    save_checkpoint(
        checkpoint_dir,
        step,
        orig_model.state_dict(),
        [opt.state_dict() for opt in optimizers],
        {...}
    )
```

**MLX:** No checkpoint saving implemented.

**Impact:** Can't save trained models, can't resume training.

---

## Architecture Differences (Addressed)

### ✅ RoPE Implementation - FIXED

**Status:** The gradient issue has been resolved with manual RoPE computation.

**PyTorch:** Uses `apply_rotary_pos_emb()` from [nanochat/gpt.py](nanochat/gpt.py)

**MLX (Original Issue):** Used `mx.fast.rope()` which doesn't support gradients.

**MLX (Fixed - [base_train_mlx.py:29-82](scripts/base_train_mlx.py#L29-L82)):**
Now uses custom `apply_rotary_emb_grad_safe()` that manually computes RoPE:
- Computes frequencies manually
- Applies rotation using cos/sin
- Supports gradient flow
- Matches PyTorch's negated frequency approach

---

### ✅ Weight Initialization - FIXED

**Status:** Now matches PyTorch initialization.

**PyTorch ([nanochat/gpt.py](nanochat/gpt.py)):** Custom init in `init_weights()`:
- Linear: `std = 1/√fan_in * min(1, √(fan_out/fan_in))`
- Embedding: `std = 1.0`
- Zero out c_proj and lm_head

**MLX ([base_train_mlx.py:303-346](scripts/base_train_mlx.py#L303-L346)):** Matches PyTorch init exactly.

---

## Minor Differences

### 7. **No Distributed Training Support** ℹ️ INFO

**PyTorch:** Supports DDP with `torchrun --nproc_per_node=8`

**MLX:** Single device only (no `ddp_world_size`, no multi-GPU)

**Impact:** Limited to single M-series GPU, slower training on large models.

---

### 8. **No Mixed Precision / Autocast** ℹ️ INFO

**PyTorch ([base_train.py:76](scripts/base_train.py#L76)):**
```python
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()
```

**MLX:** No explicit dtype casting (MLX uses float32 by default).

**Impact:** May be slower, higher memory usage. MLX could benefit from float16/bfloat16.

---

### 9. **No Model Compilation** ℹ️ INFO

**PyTorch ([base_train.py:149](scripts/base_train.py#L149)):**
```python
model = torch.compile(model, dynamic=False)
```

**MLX:** No compilation (MLX has its own JIT, but not explicitly enabled here).

**Impact:** May miss some optimizations.

---

### 10. **Different MFU Calculation** ℹ️ INFO

**PyTorch ([base_train.py:342-343](scripts/base_train.py#L342-L343)):**
```python
promised_flops_per_sec_h100 = 989e12 * ddp_world_size
mfu = 100 * flops_per_sec / promised_flops_per_sec_h100
```

**MLX:** No MFU calculation (would need M-series chip TFLOPS spec).

**Impact:** Can't measure hardware utilization.

---

## Loss Computation Differences

### 11. **Loss Computation - Likely Correct** ✅ LOW PRIORITY

**PyTorch ([nanochat/gpt.py](nanochat/gpt.py)):**
```python
# In GPT.forward()
logits = self.lm_head(x)
logits = softcap(logits)  # Softcap before loss
loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
```

**MLX ([base_train_mlx.py:459-485](scripts/base_train_mlx.py#L459-L485)):**
```python
def compute_loss(model, inputs, targets):
    logits = model(inputs)  # Softcap applied in model.__call__
    # Manual cross-entropy with log_softmax
    logits_flat = logits.reshape(-1, V)
    targets_flat = targets.reshape(-1)
    max_logits = mx.max(logits_flat, axis=-1, keepdims=True)
    exp_logits = mx.exp(logits_flat - max_logits)
    log_probs = (logits_flat - max_logits) - mx.log(mx.sum(exp_logits, axis=-1, keepdims=True))
    target_log_probs = log_probs[batch_indices, targets_flat]
    loss = -mx.mean(target_log_probs)
```

**Analysis:** Both apply softcap then cross-entropy. MLX uses numerically stable log_softmax. Should be equivalent.

**Potential issue:** Verify softcap is applied in model forward pass (yes, it is - line 289).

---

## Data Loading Differences

### 12. **Data Loader - Likely Correct** ✅ LOW PRIORITY

**PyTorch ([base_train.py:187](scripts/base_train.py#L187)):**
```python
train_loader = tokenizing_distributed_data_loader(device_batch_size, max_seq_len, split="train", device=device)
```

**MLX ([base_train_mlx.py:424-453](scripts/base_train_mlx.py#L424-L453)):**
Custom `mlx_data_loader()` that:
- Uses `parquets_iter_batched()` (same as PyTorch)
- Tokenizes with same tokenizer
- Converts to MLX arrays

**Analysis:** Logic looks equivalent. Both use same data source and tokenization.

---

## Gradient Accumulation Differences

### 13. **Gradient Accumulation - Implementation Differs** ⚠️ MEDIUM

**PyTorch ([base_train.py:308-314](scripts/base_train.py#L308-L314)):**
```python
for micro_step in range(grad_accum_steps):
    with autocast_ctx:
        loss = model(x, y)
    train_loss = loss.detach()
    loss = loss / grad_accum_steps  # Scale loss BEFORE backward
    loss.backward()
    x, y = next(train_loader)
```

**MLX ([base_train_mlx.py:650-666](scripts/base_train_mlx.py#L650-L666)):**
```python
for micro_step in range(grad_accum_steps):
    loss, grads = loss_and_grad_fn(model, x, y)
    accumulated_loss += float(loss)

    # Accumulate gradients
    if accumulated_grads is None:
        accumulated_grads = grads
    else:
        accumulated_grads = tree_map(lambda a, b: a + b, accumulated_grads, grads)

    if micro_step < grad_accum_steps - 1:
        x, y = next(train_loader)

# Average gradients AFTER accumulation
accumulated_grads = tree_map(lambda g: g / grad_accum_steps, accumulated_grads)
```

**Difference:**
- PyTorch: Scales loss before backward (so grads are pre-scaled)
- MLX: Accumulates unscaled grads, then divides

**Analysis:** Mathematically equivalent, but verify no numerical differences.

---

## Summary of Issues by Priority

### 🔴 CRITICAL (Fix Immediately)
1. **Single optimizer instead of AdamW + Muon** - Core training algorithm wrong
2. **Incorrect LR scaling** - All params get wrong learning rates

### 🟡 HIGH (Fix Soon)
3. **Missing Muon momentum scheduling**
4. **Validation not implemented** - Can't measure progress

### 🟠 MEDIUM (Fix When Possible)
5. **CORE metric disabled**
6. **No checkpoint saving**
13. **Gradient accumulation differs** (verify equivalence)

### 🔵 LOW / INFO (Future Improvements)
7. **No distributed training**
8. **No mixed precision**
9. **No model compilation**
10. **No MFU calculation**
11. **Loss computation** (verify correctness)
12. **Data loader** (verify equivalence)

---

## Root Cause of High Loss

**Primary Issue:** Single AdamW optimizer with averaged LR instead of separate AdamW + Muon.

**Why this breaks training:**
- Matrix layers (attention Q/K/V/proj, MLP fc/proj) are the core learning components
- Muon is specifically designed for matrix parameters (preconditioned gradient descent)
- Using AdamW instead of Muon → wrong update direction → poor learning
- Wrong LR (0.1 instead of 0.02 for matrices) → overshooting or undershooting

**Evidence:**
- Loss ~17 vs PyTorch ~7.5 after 100 steps
- Model produces gibberish when sampling
- Gradients are computing (RoPE fixed), but updates are ineffective

---

## Next Steps (Priority Order)

1. **Port Muon optimizer to MLX** (or find MLX equivalent)
2. **Implement separate param groups** (embedding, lm_head, matrix)
3. **Fix LR scheduling** per param group
4. **Add Muon momentum scheduling**
5. **Implement validation** (`evaluate_bpb` for MLX)
6. **Add checkpoint saving/loading**
7. Test training for 100 steps and compare loss to PyTorch
8. Implement CORE metric evaluation
9. Add distributed training support (if needed)
10. Add mixed precision / bfloat16 support

---

## Code Locations

### PyTorch Reference
- [base_train.py](scripts/base_train.py) - Main training script
- [nanochat/gpt.py](nanochat/gpt.py) - Model definition with `setup_optimizers()`

### MLX Port
- [base_train_mlx.py](scripts/base_train_mlx.py) - MLX training script (current)
- [nanochat/gpt_mlx_wrapper.py](nanochat/gpt_mlx_wrapper.py) - MLX model wrapper (if exists)

### MLX-LM Model
- `mlx_lm/models/nanochat.py` - Model definition (external package)

---

## Questions to Investigate

1. **Does MLX have a Muon optimizer?**
   - Check `mlx.optimizers` module
   - May need to port from PyTorch Muon implementation

2. **Can MLX optimizers handle param groups?**
   - PyTorch: `optimizer.param_groups` with different LRs
   - MLX: Need to verify if this is supported

3. **Is gradient accumulation mathematically equivalent?**
   - PyTorch: scale loss before backward
   - MLX: accumulate grads then scale
   - Should be same, but verify numerically

4. **Are there MLX-specific numerical differences?**
   - Float32 vs bfloat16
   - Different math kernel implementations
   - RoPE frequency computation

---

## Conclusion

The MLX port is **not training correctly** due to **optimizer setup issues**. The RoPE and initialization are now fixed, but the single AdamW optimizer (instead of AdamW + Muon) and incorrect LR scaling prevent the model from learning.

**Fix priority:** Implement separate optimizers with correct LR scheduling first, then add validation and checkpointing.
