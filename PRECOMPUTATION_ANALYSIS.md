# Precomputation Optimization Analysis

## Summary

Several expensive computations are being recomputed every forward pass that could be **precomputed once** during initialization. These optimizations could provide **5-15% speedup** with minimal code changes.

---

## 🎯 High-Priority Optimizations

### 1. RoPE Frequencies (Currently Partial)

**Current state** (lines 105-108):
```python
# In AttentionGradSafe.__init__:
half_D = self.head_dim // 2
self._rope_freqs = -mx.exp(
    mx.arange(0.0, half_D, dtype=mx.float32)
    * (math.log(self.rope_theta) / half_D)
)
```

**Problem**: Good start, but then in `apply_rotary_emb_grad_safe` (lines 40-46) we **recompute the same thing**:
```python
def apply_rotary_emb_grad_safe(x, offset=0, base=10000.0, freqs=None):
    if freqs is None:  # ← This fallback is being used!
        half_D = x.shape[-1] // 2
        freqs = -mx.exp(
            mx.arange(0.0, half_D, dtype=mx.float32) * (math.log(base) / half_D)
        )
    freqs = mx.stop_gradient(freqs)
```

**Issue**: The `freqs` parameter IS being passed (line 128, 131), but we're **recomputing positions, angles, cos, sin every time**.

**What can be precomputed**:

```python
# Precompute cos/sin lookup tables for max_seq_len positions
# In AttentionGradSafe.__init__:
max_positions = args.max_position_embeddings
half_D = self.head_dim // 2

# Precompute frequencies
self._rope_freqs = -mx.exp(
    mx.arange(0.0, half_D, dtype=mx.float32)
    * (math.log(self.rope_theta) / half_D)
)

# Precompute all position angles [0..max_positions]
positions = mx.arange(0.0, max_positions, dtype=mx.float32)  # (max_seq,)
angles = positions[:, None] * self._rope_freqs[None, :]  # (max_seq, half_D)

# Precompute cos/sin tables
self._rope_cos = mx.cos(angles)[None, None, :, :]  # (1, 1, max_seq, half_D)
self._rope_sin = mx.sin(angles)[None, None, :, :]  # (1, 1, max_seq, half_D)
self._rope_cos = mx.stop_gradient(self._rope_cos)
self._rope_sin = mx.stop_gradient(self._rope_sin)
```

Then in forward pass (much faster):
```python
# Just slice the precomputed tables
offset_int = int(offset) if isinstance(offset, mx.array) else offset
L = x.shape[2]
cos_vals = self._rope_cos[:, :, offset_int:offset_int+L, :]
sin_vals = self._rope_sin[:, :, offset_int:offset_int+L, :]

# Apply rotation (no trig computation!)
x1 = x[..., :half_D]
x2 = x[..., half_D:]
rotated_x1 = x1 * cos_vals - x2 * sin_vals
rotated_x2 = x1 * sin_vals + x2 * cos_vals
return mx.concatenate([rotated_x1, rotated_x2], axis=-1)
```

**Savings**:
- Eliminates per-forward: position generation, angle multiplication, cos(), sin()
- Estimated speedup: **3-5%** (RoPE is called 2x per layer per forward)

---

### 2. Attention Mask (Currently Recomputed)

**Current state** (line 341):
```python
# In GradSafeNanoChatModel.__call__:
mask = create_attention_mask(h, cache[0])
```

**Problem**: This creates a causal mask **every forward pass**, even though:
- Mask shape is always (seq_len, seq_len)
- Mask pattern is always lower-triangular (causal)
- Only the size changes

**What can be precomputed**:

```python
# In GradSafeNanoChatModel.__init__:
max_seq_len = args.max_position_embeddings

# Precompute causal mask for max_seq_len
# Shape: (1, 1, max_seq, max_seq)
# True where we can attend, False where we can't
positions = mx.arange(max_seq_len)
self._causal_mask_full = positions[:, None] >= positions[None, :]
self._causal_mask_full = self._causal_mask_full[None, None, :, :]  # (1, 1, max_seq, max_seq)
self._causal_mask_full = mx.stop_gradient(self._causal_mask_full)
```

Then in forward:
```python
# Just slice the precomputed mask
L = h.shape[1]  # Current sequence length
mask = self._causal_mask_full[:, :, :L, :L]
```

**Savings**:
- Eliminates per-forward: arange(), broadcasting, comparison ops
- Estimated speedup: **1-2%** (called once per forward)

---

### 3. Scale Factor in Attention (Trivial)

**Current state** (line 95):
```python
self.scale = self.head_dim**-0.5
```

**Already optimal!** ✅ This is correctly precomputed in `__init__`.

---

### 4. Softcap Computation (Potentially Redundant)

**Current state** (line 363):
```python
# In GradSafeModel.__call__:
logits = softcap(logits)
```

**Question**: What does `softcap` do? Let me check if it has constants that could be precomputed.

From `mlx_lm.models.nanochat`:
```python
def softcap(x, cap=30.0):
    return cap * mx.tanh(x / cap)
```

**Analysis**:
- `cap=30.0` is a constant
- `1/cap` could be precomputed as `self._softcap_inv = 1.0 / 30.0`
- Then: `cap * mx.tanh(x * self._softcap_inv)`

**Savings**: Minimal (1 division saved per forward), **not worth it**.

---

## 🔧 Medium-Priority Optimizations

### 5. Head Dimension Calculations

**Current state**:
```python
# In multiple places:
queries.reshape(B, L, self.num_heads, self.head_dim)
```

**What can be precomputed**:
```python
# In AttentionGradSafe.__init__:
self._qkv_reshape_dims = (self.num_heads, self.head_dim)
self._kv_reshape_dims = (self.num_kv_heads, self.head_dim)
```

**Savings**: Negligible (Python integer tuples are cheap). **Skip this**.

---

### 6. RMSNorm Epsilon

**Current state**:
```python
# rms_norm is called with default epsilon every time
queries = rms_norm(queries)
keys = rms_norm(keys)
```

If `rms_norm` has configurable epsilon, it's being set every call. Check the implementation.

**Likely already optimal** (functional API). **Skip**.

---

## 📊 Estimated Performance Impact

| Optimization | Speedup | Complexity | Priority |
|--------------|---------|------------|----------|
| **RoPE cos/sin precomputation** | 3-5% | Medium | ⭐⭐⭐ HIGH |
| **Attention mask precomputation** | 1-2% | Low | ⭐⭐ MEDIUM |
| Softcap constant | <0.1% | Low | ❌ SKIP |
| Head dim tuples | <0.1% | Low | ❌ SKIP |

**Combined potential speedup: 4-7%**

---

## 🚀 Implementation Plan

### Step 1: RoPE Precomputation (Highest Impact)

**Modify `AttentionGradSafe.__init__`** (lines 88-108):

```python
class AttentionGradSafe(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.num_heads = args.num_attention_heads
        self.num_kv_heads = args.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.scale = self.head_dim**-0.5
        self.rope_theta = args.rope_theta

        self.c_q = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        # Precompute RoPE cos/sin lookup tables
        max_seq_len = args.max_position_embeddings
        half_D = self.head_dim // 2

        # Frequencies (already had this)
        freqs = -mx.exp(
            mx.arange(0.0, half_D, dtype=mx.float32)
            * (math.log(self.rope_theta) / half_D)
        )

        # Precompute angles for all positions [0..max_seq_len]
        positions = mx.arange(0.0, max_seq_len, dtype=mx.float32)
        angles = positions[:, None] * freqs[None, :]  # (max_seq, half_D)

        # Precompute cos/sin tables (shape: 1, 1, max_seq, half_D)
        self._rope_cos = mx.cos(angles)[None, None, :, :]
        self._rope_sin = mx.sin(angles)[None, None, :, :]
        self._rope_cos = mx.stop_gradient(self._rope_cos)
        self._rope_sin = mx.stop_gradient(self._rope_sin)
```

**Modify forward pass** (lines 125-132):

```python
# Apply RoPE using precomputed cos/sin tables
offset = int(cache.offset if cache is not None else 0)
queries = self._apply_rope_fast(queries, offset)
keys = self._apply_rope_fast(keys, offset)
```

**Add helper method**:

```python
def _apply_rope_fast(self, x, offset):
    """Apply RoPE using precomputed cos/sin tables."""
    B, H, L, D = x.shape
    half_D = D // 2

    # Slice precomputed tables
    cos_vals = self._rope_cos[:, :, offset:offset+L, :]
    sin_vals = self._rope_sin[:, :, offset:offset+L, :]

    # Split and rotate
    x1 = x[..., :half_D]
    x2 = x[..., half_D:]
    rotated_x1 = x1 * cos_vals - x2 * sin_vals
    rotated_x2 = x1 * sin_vals + x2 * cos_vals

    return mx.concatenate([rotated_x1, rotated_x2], axis=-1)
```

---

### Step 2: Attention Mask Precomputation

**Modify `GradSafeNanoChatModel.__init__`** (lines 324-330):

```python
class GradSafeNanoChatModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.wte = nn.Embedding(args.vocab_size, args.hidden_size)
        self.h = [TransformerBlockGradSafe(args) for _ in range(args.num_hidden_layers)]

        # Precompute causal attention mask
        max_seq = args.max_position_embeddings
        positions = mx.arange(max_seq)
        causal_mask = positions[:, None] >= positions[None, :]
        self._causal_mask = causal_mask[None, None, :, :]  # (1, 1, max_seq, max_seq)
        self._causal_mask = mx.stop_gradient(self._causal_mask)
```

**Modify forward** (line 341):

```python
# Slice precomputed mask instead of creating it
L = h.shape[1]
mask = self._causal_mask[:, :, :L, :L]
```

---

## ⚠️ Caveats and Considerations

### Memory Trade-off

Precomputing cos/sin tables:
```python
# Memory: (1, 1, max_seq_len, head_dim/2) × 2 (cos + sin) × 2 bytes (bf16)
# Example: max_seq=1024, head_dim=128
# = (1 × 1 × 1024 × 64) × 2 × 2 = 262 KB per attention layer
# For 20 layers: 5.2 MB total

# Attention mask:
# (1, 1, max_seq, max_seq) × 1 byte (bool) = 1024×1024 = 1 MB
```

**Total memory overhead: ~6 MB** (negligible compared to 3-10 GB model memory)

### Cache Invalidation

If you ever change `max_seq_len` dynamically, you'd need to rebuild these tables. For fixed-size training (current setup), this is not an issue.

### Compilation Benefits

Precomputed constants are **even better with `mx.compile()`**:
- Lookup operations (slicing) are very fast
- No dynamic trig computation to compile
- Better cache locality

---

## 🎓 Lessons

1. **Trig functions are expensive** - cos/sin are slow on GPU, precompute when possible
2. **Indexing is cheap** - Slicing precomputed arrays is much faster than recomputing
3. **Fixed-size training enables precomputation** - Knowing max_seq_len ahead of time is key
4. **Memory is cheap, compute is expensive** - 6 MB for 4-7% speedup is a great trade

---

## 📁 Files to Modify

- **[scripts/base_train_mlx.py](scripts/base_train_mlx.py)**
  - Lines 86-148: `AttentionGradSafe` class
  - Lines 324-347: `GradSafeNanoChatModel` class
  - Lines 35-83: Remove/simplify `apply_rotary_emb_grad_safe` (or keep as fallback)

---

**Recommendation**: Implement RoPE precomputation first (3-5% gain), then attention mask (1-2% gain). Total expected speedup: **4-7%** with minimal risk.
