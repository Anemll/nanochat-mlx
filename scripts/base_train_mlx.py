"""
MLX version of base_train.py - pretraining with MLX-LM nanochat model.

This is an iterative port starting with AdamW optimizer only.
Goal: Match PyTorch results for 100 steps.

Run as:
python -m scripts.base_train_mlx --depth=4 --max_seq_len=1024 --device_batch_size=1 --total_batch_size=1024 --num_iterations=100
"""

import os
import time
import math
from collections import deque

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

# Import MLX-LM nanochat model
try:
    from mlx_lm.models.nanochat import Model as MLXNanoChatModel, ModelArgs as MLXModelArgs
    from mlx_lm.models.nanochat import Attention as MLXAttention
    from mlx_optimizers import Muon
except ImportError:
    raise ImportError("Please install MLX dependencies: uv sync --extra mlx (or pip install mlx mlx-lm mlx-optimizers)")

# Custom RoPE implementation that supports gradients (manual computation, no mx.fast.rope)
def apply_rotary_emb_grad_safe(x, offset, base=10000.0, freqs=None):
    """
    Apply RoPE manually to support gradients.
    Does NOT use mx.fast.rope - computes RoPE manually so gradients can flow.
    """
    head_dim = x.shape[-1]
    half_D = head_dim // 2
    
    # Compute frequencies if not provided
    if freqs is None:
        # Compute negated frequencies (matching MLX-LM's approach)
        freqs = -mx.exp(
            mx.arange(0.0, half_D, dtype=mx.float32) * (math.log(base) / half_D)
        )
    
    # Stop gradients on frequencies (they're constants, not trainable)
    freqs = mx.stop_gradient(freqs)
    
    # Convert offset to int if it's an array (for indexing)
    if isinstance(offset, mx.array):
        offset = int(offset.item())
    elif not isinstance(offset, int):
        offset = int(offset)
    
    # Get sequence length and batch/head dimensions
    B, H, L, D = x.shape
    
    # Create position indices: [offset, offset+1, ..., offset+L-1]
    positions = mx.arange(float(offset), float(offset + L), dtype=mx.float32)  # (L,)
    
    # Compute angles: positions[:, None] * freqs[None, :] -> (L, half_D)
    angles = positions[:, None] * freqs[None, :]  # (L, half_D)
    
    # Compute cos and sin
    cos_vals = mx.cos(angles)  # (L, half_D)
    sin_vals = mx.sin(angles)  # (L, half_D)
    
    # Expand to match x shape: (1, 1, L, half_D)
    cos_vals = cos_vals[None, None, :, :]  # (1, 1, L, half_D)
    sin_vals = sin_vals[None, None, :, :]  # (1, 1, L, half_D)
    
    # Split x into two halves
    x1 = x[..., :half_D]  # (B, H, L, half_D)
    x2 = x[..., half_D:]  # (B, H, L, half_D)
    
    # Apply rotation: [x1*cos - x2*sin, x1*sin + x2*cos]
    # This matches the negated frequency approach (rotating in opposite direction)
    rotated_x1 = x1 * cos_vals - x2 * sin_vals
    rotated_x2 = x1 * sin_vals + x2 * cos_vals
    
    # Concatenate back
    rotated = mx.concatenate([rotated_x1, rotated_x2], axis=-1)
    
    return rotated

# Custom Attention layer with gradient-safe RoPE
class AttentionGradSafe(nn.Module):
    """Attention layer with gradient-safe RoPE."""
    def __init__(self, args):
        super().__init__()
        # Copy all attributes from MLXAttention
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

        # Precompute RoPE cos/sin lookup tables for maximum sequence length
        # This eliminates expensive trig computation from the forward pass (3-5% speedup)
        max_seq_len = args.max_position_embeddings
        half_D = self.head_dim // 2

        # Compute negated frequencies (matching MLX-LM's convention)
        freqs = -mx.exp(
            mx.arange(0.0, half_D, dtype=mx.float32)
            * (math.log(self.rope_theta) / half_D)
        )

        # Precompute angles for all positions [0, 1, 2, ..., max_seq_len-1]
        positions = mx.arange(0.0, max_seq_len, dtype=mx.float32)  # (max_seq,)
        angles = positions[:, None] * freqs[None, :]  # (max_seq, half_D)

        # Precompute cos/sin lookup tables
        # Shape: (1, 1, max_seq_len, half_D) for broadcasting with (B, H, L, half_D)
        self._rope_cos = mx.cos(angles)[None, None, :, :]
        self._rope_sin = mx.sin(angles)[None, None, :, :]

        # Stop gradients (these are constants, not trainable parameters)
        self._rope_cos = mx.stop_gradient(self._rope_cos)
        self._rope_sin = mx.stop_gradient(self._rope_sin)

    def _apply_rope_fast(self, x, offset):
        """Apply RoPE using precomputed cos/sin lookup tables.

        Args:
            x: Input tensor (B, H, L, D)
            offset: Position offset for sequence (int)

        Returns:
            Rotated tensor (B, H, L, D)
        """
        B, H, L, D = x.shape
        half_D = D // 2

        # Convert offset to int if needed
        if isinstance(offset, mx.array):
            offset = int(offset.item())
        elif not isinstance(offset, int):
            offset = int(offset)

        # Slice precomputed cos/sin tables (no trig computation!)
        cos_vals = self._rope_cos[:, :, offset:offset+L, :]  # (1, 1, L, half_D)
        sin_vals = self._rope_sin[:, :, offset:offset+L, :]  # (1, 1, L, half_D)

        # Split x into two halves
        x1 = x[..., :half_D]  # (B, H, L, half_D)
        x2 = x[..., half_D:]  # (B, H, L, half_D)

        # Apply rotation: [x1*cos - x2*sin, x1*sin + x2*cos]
        rotated_x1 = x1 * cos_vals - x2 * sin_vals
        rotated_x2 = x1 * sin_vals + x2 * cos_vals

        # Concatenate back
        return mx.concatenate([rotated_x1, rotated_x2], axis=-1)

    def __call__(self, x, mask=None, cache=None):
        from mlx_lm.models.base import scaled_dot_product_attention
        from mlx_lm.models.nanochat import rms_norm
        
        B, L, _ = x.shape
        
        queries = self.c_q(x)
        keys = self.c_k(x)
        values = self.c_v(x)
        
        # Reshape to (B, L, H, D) then transpose to (B, H, L, D)
        queries = queries.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Apply RoPE using precomputed cos/sin tables (fast path - no trig!)
        offset = cache.offset if cache is not None else 0
        queries = self._apply_rope_fast(queries, offset)
        keys = self._apply_rope_fast(keys, offset)
        
        # QK norm (critical feature of nanochat!)
        queries = rms_norm(queries)
        keys = rms_norm(keys)
        
        # Handle KV cache after transpose
        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)
        
        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        
        # Reshape back
        output = output.transpose(0, 2, 1, 3).reshape(B, L, self.hidden_size)
        return self.c_proj(output)

import wandb

from nanochat.common import print0, DummyWandb, print_banner, get_base_dir
from nanochat.dataset import parquets_iter_batched
from nanochat.tokenizer import get_tokenizer, get_token_bytes

print_banner()

# -----------------------------------------------------------------------------
# Evaluation function for MLX (bits per byte)
def evaluate_bpb_mlx(model, batches, steps, token_bytes_np):
    """
    MLX version of evaluate_bpb.
    Calculates bits per byte (bpb), a tokenization-independent metric.

    Args:
        model: MLX model
        batches: Iterator of (inputs, targets) batches
        steps: Number of batches to evaluate
        token_bytes_np: NumPy array of shape (vocab_size,) with byte counts per token

    Returns:
        bpb: Bits per byte metric
    """
    import math

    # Convert token_bytes to MLX array
    token_bytes_mx = mx.array(token_bytes_np)

    total_nats = 0.0
    total_bytes = 0
    batch_iter = iter(batches)

    for _ in range(steps):
        inputs, targets = next(batch_iter)

        # Compute loss per token (without reduction)
        logits = model(inputs)
        B, T, V = logits.shape
        logits_flat = logits.reshape(-1, V)
        targets_flat = targets.reshape(-1).astype(mx.int32)

        # Compute per-token losses (manual cross-entropy without reduction)
        max_logits = mx.max(logits_flat, axis=-1, keepdims=True)
        exp_logits = mx.exp(logits_flat - max_logits)
        log_probs = (logits_flat - max_logits) - mx.log(mx.sum(exp_logits, axis=-1, keepdims=True))

        # Gather log probs for target tokens
        batch_size = log_probs.shape[0]
        batch_indices = mx.arange(batch_size)
        target_log_probs = log_probs[batch_indices, targets_flat]
        loss2d = -target_log_probs  # Per-token losses

        # Map targets to byte counts
        # Handle potential negative indices (ignore_index)
        valid = targets_flat >= 0
        targets_safe = mx.where(valid, targets_flat, mx.zeros_like(targets_flat))

        # Get byte counts for each token
        num_bytes2d = token_bytes_mx[targets_safe]
        # Zero out bytes for invalid targets
        num_bytes2d = mx.where(valid, num_bytes2d, mx.zeros_like(num_bytes2d))

        # Accumulate nats and bytes (only for tokens with byte count > 0)
        mask = num_bytes2d > 0
        total_nats += float(mx.sum(loss2d * mask))
        total_bytes += int(mx.sum(num_bytes2d))

    # Calculate bpb
    if total_bytes == 0:
        return float('inf')

    bpb = total_nats / (math.log(2) * total_bytes)
    return bpb

# -----------------------------------------------------------------------------
# User settings (matching base_train.py)
run = "dummy"  # wandb run name
depth = 20  # model depth
max_seq_len = 2048  # max context length
num_iterations = -1  # explicit number of steps (-1 = disable)
target_flops = -1.0  # calculate from target FLOPs (-1 = disable)
target_param_data_ratio = 20  # Chinchilla ratio (-1 = disable)
device_batch_size = 32  # per-device batch size
use_bfloat16 = True  # use bfloat16 mixed precision (2x memory reduction)
total_batch_size = 524288  # total desired batch size
embedding_lr = 0.2  # learning rate for embedding
unembedding_lr = 0.004  # learning rate for lm_head
matrix_lr = 0.02  # learning rate for matrix parameters (Muon - not yet implemented)
weight_decay = 0.0  # weight decay
grad_clip = 1.0  # gradient clipping (0.0 = disabled)
warmup_ratio = 0.0  # LR warmup ratio
warmdown_ratio = 0.2  # LR warmdown ratio
final_lr_frac = 0.0  # final LR fraction
eval_every = 250  # evaluate every N steps
eval_tokens = 20 * 524288  # tokens for evaluation
core_metric_every = -1  # disable CORE metric for now
sample_every = 2000  # every how many steps to sample from the model (-1 = disable)
save_every = -1  # save checkpoint every N steps (-1 = only save at end)
model_tag = ""  # model tag for checkpoint
resume_from_checkpoint = ""  # checkpoint path
resume_step = -1  # step to resume from

# Apply configurator overrides
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join('nanochat', 'configurator.py')).read())
user_config = {k: globals()[k] for k in config_keys}

# -----------------------------------------------------------------------------
# Wandb init
use_dummy_wandb = run == "dummy"
if not use_dummy_wandb:
    print0(f"wandb run name: '{run}'")
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat", name=run, config=user_config)

# Tokenizer
tokenizer = get_tokenizer()
token_bytes = get_token_bytes()  # Get token bytes for validation metric
vocab_size = tokenizer.get_vocab_size()
print0(f"Vocab size: {vocab_size:,}")

# Model architecture (matching PyTorch version)
num_layers = depth
model_dim = depth * 64  # aspect ratio 64
num_heads = max(1, (model_dim + 127) // 128)  # head dim 128
num_kv_heads = num_heads  # 1:1 GQA ratio
print0(f"num_layers: {num_layers}")
print0(f"model_dim: {model_dim}")
print0(f"num_heads: {num_heads}")
print0(f"num_kv_heads: {num_kv_heads}")

# Gradient accumulation
tokens_per_fwdbwd = device_batch_size * max_seq_len
grad_accum_steps = total_batch_size // tokens_per_fwdbwd
print0(f"Tokens / micro-batch: {tokens_per_fwdbwd:,}")
print0(f"Total batch size {total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")

# Memory logging configuration (set > 0 to enable periodic logging)
MEMLOG_EVERY = 10  # Log memory every N steps (0 = disabled)

# -----------------------------------------------------------------------------
# Initialize MLX Model
mlx_args = MLXModelArgs(
    hidden_size=model_dim,
    num_hidden_layers=num_layers,
    num_attention_heads=num_heads,
    num_key_value_heads=num_kv_heads,
    vocab_size=vocab_size,
    max_position_embeddings=max_seq_len,
    intermediate_size=4 * model_dim,  # 4 * hidden_size
    rope_theta=10000.0,
)

# Create model with gradient-safe attention layers
# We need to replace the Attention layers with our gradient-safe version
def create_grad_safe_model(args):
    """Create nanochat model with gradient-safe RoPE."""
    from mlx_lm.models.nanochat import MLP, TransformerBlock as MLXTransformerBlock
    from mlx_lm.models.nanochat import rms_norm
    from functools import partial
    
    # Create a custom TransformerBlock that uses our gradient-safe attention
    class TransformerBlockGradSafe(nn.Module):
        def __init__(self, args):
            super().__init__()
            self.attn = AttentionGradSafe(args)
            self.mlp = MLP(args)
        
        def __call__(self, x, mask=None, cache=None):
            # Pre-norm architecture with functional RMSNorm
            h = x + self.attn(rms_norm(x), mask=mask, cache=cache)
            out = h + self.mlp(rms_norm(h))
            return out
    
    class GradSafeNanoChatModel(nn.Module):
        def __init__(self, args):
            super().__init__()
            self.args = args
            self.wte = nn.Embedding(args.vocab_size, args.hidden_size)
            # Use gradient-safe transformer blocks
            self.h = [TransformerBlockGradSafe(args) for _ in range(args.num_hidden_layers)]

            # Precompute causal attention mask for maximum sequence length
            # This eliminates mask creation from the forward pass (1-2% speedup)
            max_seq = args.max_position_embeddings
            positions = mx.arange(max_seq)
            # Causal mask: can attend to current and previous positions
            # positions[:, None] >= positions[None, :] creates lower-triangular pattern
            causal_mask = positions[:, None] >= positions[None, :]  # (max_seq, max_seq)
            # Add batch and head dimensions: (1, 1, max_seq, max_seq)
            self._causal_mask = causal_mask[None, None, :, :]
            self._causal_mask = mx.stop_gradient(self._causal_mask)
        
        def __call__(self, inputs, cache=None):
            h = self.wte(inputs)
            h = rms_norm(h)

            if cache is None:
                cache = [None] * len(self.h)

            # Use precomputed causal mask (just slice to current sequence length)
            L = h.shape[1]  # Current sequence length
            mask = self._causal_mask[:, :, :L, :L]

            for layer, c in zip(self.h, cache):
                h = layer(h, mask=mask, cache=c)

            h = rms_norm(h)
            return h
    
    # Create the full model
    class GradSafeModel(nn.Module):
        def __init__(self, args):
            super().__init__()
            self.args = args
            self.model_type = args.model_type
            self.transformer = GradSafeNanoChatModel(args)
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        
        def __call__(self, inputs, cache=None):
            from mlx_lm.models.nanochat import softcap
            
            out = self.transformer(inputs, cache=cache)
            logits = self.lm_head(out)
            logits = softcap(logits)
            return logits
        
        @property
        def layers(self):
            return self.transformer.h
    
    return GradSafeModel(args)

# Create model with gradient-safe RoPE (no mx.fast.rope in forward pass)
model = create_grad_safe_model(mlx_args)
print0("Created model with gradient-safe RoPE (manual computation, no mx.fast.rope)")

# Set model dtype BEFORE initialization to avoid FP32 intermediates
# This is the proper MLX way to set parameter dtypes
MODEL_DTYPE = mx.bfloat16 if use_bfloat16 else mx.float32
model.set_dtype(MODEL_DTYPE)
print0(f"Model parameters dtype set to {MODEL_DTYPE}")

# Apply PyTorch-style weight initialization to match base_train.py
def init_weights_pytorch_style(model):
    """Initialize weights to match PyTorch init_weights() from nanochat/gpt.py

    IMPORTANT: Uses MLX RNG directly with weight.dtype to avoid creating FP32 intermediates.
    This prevents memory bloat when using bfloat16/float16 training.
    """
    import math

    def init_linear_weight(weight):
        """Initialize Linear layer weight: std = 1/√fan_in * min(1, √(fan_out/fan_in))"""
        fan_out, fan_in = weight.shape
        std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
        # Use MLX random with weight's dtype (not NumPy with explicit fp32 cast!)
        weight_array = mx.random.normal(shape=weight.shape, dtype=weight.dtype, loc=0.0, scale=std)
        return weight_array

    def init_embedding_weight(weight):
        """Initialize Embedding weight: std=1.0"""
        # Use MLX random with weight's dtype (not NumPy with explicit fp32 cast!)
        weight_array = mx.random.normal(shape=weight.shape, dtype=weight.dtype, loc=0.0, scale=1.0)
        return weight_array
    
    # Initialize embedding
    model.transformer.wte.weight = init_embedding_weight(model.transformer.wte.weight)
    
    # Initialize transformer blocks
    for layer in model.transformer.h:
        # Attention layers
        if hasattr(layer.attn, 'c_q'):
            layer.attn.c_q.weight = init_linear_weight(layer.attn.c_q.weight)
        if hasattr(layer.attn, 'c_k'):
            layer.attn.c_k.weight = init_linear_weight(layer.attn.c_k.weight)
        if hasattr(layer.attn, 'c_v'):
            layer.attn.c_v.weight = init_linear_weight(layer.attn.c_v.weight)
        # Zero out c_proj (matching PyTorch)
        layer.attn.c_proj.weight = mx.zeros_like(layer.attn.c_proj.weight)
        
        # MLP layers
        if hasattr(layer.mlp, 'c_fc'):
            layer.mlp.c_fc.weight = init_linear_weight(layer.mlp.c_fc.weight)
        if hasattr(layer.mlp, 'c_proj'):
            # Zero out c_proj (matching PyTorch)
            layer.mlp.c_proj.weight = mx.zeros_like(layer.mlp.c_proj.weight)
    
    # Zero out lm_head (matching PyTorch)
    model.lm_head.weight = mx.zeros_like(model.lm_head.weight)

    print0("Applied PyTorch-style weight initialization (matching base_train.py)")

# Initialize weights (they will be created in the correct dtype from model.set_dtype() above)
init_weights_pytorch_style(model)

# Verify dtype
sample_param = model.parameters()['transformer']['wte']['weight']
print0(f"  Verified parameter dtype: {sample_param.dtype}")
assert sample_param.dtype == MODEL_DTYPE, f"Parameter dtype mismatch: {sample_param.dtype} != {MODEL_DTYPE}"

# Helper functions for MLX tree operations
def tree_map(fn, tree, tree2=None):
    """Apply function to all arrays in a tree structure.
    If tree2 is provided, applies fn to corresponding pairs.
    """
    if tree2 is not None:
        # Binary operation on two trees
        if isinstance(tree, dict) and isinstance(tree2, dict):
            return {k: tree_map(fn, tree[k], tree2[k]) for k in tree.keys()}
        elif isinstance(tree, (list, tuple)) and isinstance(tree2, (list, tuple)):
            return type(tree)(tree_map(fn, a, b) for a, b in zip(tree, tree2))
        elif isinstance(tree, mx.array) and isinstance(tree2, mx.array):
            return fn(tree, tree2)
        else:
            raise ValueError("Tree structures don't match")
    else:
        # Unary operation on single tree
        if isinstance(tree, dict):
            return {k: tree_map(fn, v) for k, v in tree.items()}
        elif isinstance(tree, (list, tuple)):
            return type(tree)(tree_map(fn, v) for v in tree)
        elif isinstance(tree, mx.array):
            return fn(tree)
        else:
            return tree

def tree_flatten(tree):
    """Flatten a tree structure into a list of arrays."""
    result = []
    if isinstance(tree, dict):
        for v in tree.values():
            result.extend(tree_flatten(v))
    elif isinstance(tree, (list, tuple)):
        for v in tree:
            result.extend(tree_flatten(v))
    elif isinstance(tree, mx.array):
        result.append(tree)
    return result

def count_parameters(params):
    """Count total parameters in MLX model."""
    total = 0
    for v in params.values():
        if isinstance(v, mx.array):
            total += v.size
        elif isinstance(v, dict):
            total += count_parameters(v)
    return total

num_params = count_parameters(model.parameters())
print0(f"Number of parameters: {num_params:,}")

# Estimate FLOPs (simplified, matching PyTorch formula)
nparams_embedding = vocab_size * model_dim
num_flops_per_token = 6 * (num_params - nparams_embedding) + 12 * num_layers * num_heads * (model_dim // num_heads) * max_seq_len
print0(f"Estimated FLOPs per token: {num_flops_per_token:e}")

# Calculate iterations
assert num_iterations > 0 or target_param_data_ratio > 0 or target_flops > 0
if num_iterations > 0:
    print0(f"Using user-provided number of iterations: {num_iterations:,}")
elif target_flops > 0:
    num_iterations = round(target_flops / (num_flops_per_token * total_batch_size))
    print0(f"Calculated number of iterations from target FLOPs: {num_iterations:,}")
elif target_param_data_ratio > 0:
    target_tokens = target_param_data_ratio * num_params
    num_iterations = target_tokens // total_batch_size
    print0(f"Calculated number of iterations from target data:param ratio: {num_iterations:,}")

total_tokens = total_batch_size * num_iterations
print0(f"Total number of training tokens: {total_tokens:,}")
print0(f"Tokens : Params ratio: {total_batch_size * num_iterations / num_params:.2f}")
print0(f"Total training FLOPs estimate: {num_flops_per_token * total_tokens:e}")

# -----------------------------------------------------------------------------
# Data Loader (MLX version)
def mlx_data_loader(B, T, split):
    """MLX-compatible data loader - converts to MLX arrays."""
    assert split in ["train", "val"]
    needed_tokens = B * T + 1
    tokenizer = get_tokenizer()
    bos_token = tokenizer.get_bos_token_id()
    token_buffer = deque()
    
    def document_batches():
        while True:
            for batch in parquets_iter_batched(split=split, start=0, step=1):
                for i in range(0, len(batch), 128):
                    yield batch[i:i+128]
    
    batches = document_batches()
    
    while True:
        while len(token_buffer) < needed_tokens:
            doc_batch = next(batches)
            token_lists = tokenizer.encode(doc_batch, prepend=bos_token, num_threads=4)
            for tokens in token_lists:
                token_buffer.extend(tokens)
        
        tokens = [token_buffer.popleft() for _ in range(needed_tokens)]
        # Convert to numpy then MLX
        tokens_np = np.array(tokens, dtype=np.int32)
        inputs = mx.array(tokens_np[:-1].reshape(B, T))
        targets = mx.array(tokens_np[1:].reshape(B, T))
        yield inputs, targets

train_loader = mlx_data_loader(device_batch_size, max_seq_len, "train")
build_val_loader = lambda: mlx_data_loader(device_batch_size, max_seq_len, "val")

# -----------------------------------------------------------------------------
# Loss Function (matching PyTorch implementation)
def compute_loss(model, inputs, targets):
    """Compute cross-entropy loss matching PyTorch version."""
    logits = model(inputs)
    # Note: softcap is already applied in the model forward pass (GradSafeModel.__call__)
    # So we don't apply it again here
    
    # Flatten for cross-entropy
    B, T, V = logits.shape
    logits_flat = logits.reshape(-1, V)
    targets_flat = targets.reshape(-1).astype(mx.int32)
    
    # Manual cross-entropy: log_softmax + gather + negate
    # Use numerically stable log_softmax: log_softmax(x) = x - log(sum(exp(x)))
    # Subtract max for numerical stability
    max_logits = mx.max(logits_flat, axis=-1, keepdims=True)
    exp_logits = mx.exp(logits_flat - max_logits)
    log_probs = (logits_flat - max_logits) - mx.log(mx.sum(exp_logits, axis=-1, keepdims=True))
    
    # Gather log probs for target tokens using indexing
    batch_size = log_probs.shape[0]
    batch_indices = mx.arange(batch_size)
    target_log_probs = log_probs[batch_indices, targets_flat]
    
    # Mean loss (negative log likelihood)
    loss = -mx.mean(target_log_probs)
    
    return loss

# Create compiled loss and gradient function using MLX's module-aware gradient transform
# This uses nn.value_and_grad() which computes gradients w.r.t. model.trainable_parameters
# and can be safely compiled by capturing model.state rather than passing model as an argument
from functools import partial

# PARTIAL COMPILATION (forward + backward only - baseline for comparison)
# Capture model state for compilation (not the module itself)
# This prevents the 'dict' object is not callable error
# We specify both inputs and outputs to properly capture the state flow
state_for_compile = [model.state]

@partial(mx.compile, inputs=state_for_compile, outputs=state_for_compile)
def _loss_and_grad_compiled(inputs, targets):
    """Compiled loss and gradient computation.

    Uses nn.value_and_grad which is module-aware and computes gradients
    w.r.t. model.trainable_parameters. The model is captured in the closure
    rather than passed as a compiled argument, avoiding the pytree flattening issue.
    """
    # Module-aware gradient transform
    loss_and_grad = nn.value_and_grad(model, compute_loss)
    return loss_and_grad(model, inputs, targets)

def loss_and_grad_fn(model, inputs, targets):
    """Compute loss and gradients (compiled; model captured in closure)."""
    # Dispatch to compiled function - model is in closure, not an argument
    return _loss_and_grad_compiled(inputs, targets)

# NOTE: MLX Compilation Strategy
# We use nn.value_and_grad(model, fn) with state capture instead of mx.value_and_grad(fn, argnums=0)
# This is the official MLX pattern for compiling training graphs:
# - nn.value_and_grad works with nn.Module and computes grads w.r.t. trainable_parameters
# - Capturing [model.state] as inputs= tells mx.compile to treat it as implicit state
# - The model never becomes a function argument, so it's not flattened to a dict
#
# This gives us JIT compilation for forward + backward pass, which should significantly
# improve performance vs the uncompiled version.
#
# Reference: https://ml-explore.github.io/mlx/build/html/usage/compile.html

# FULL-GRAPH COMPILATION will be set up after optimizer initialization (see below)

# -----------------------------------------------------------------------------
# Optimizers (matching PyTorch: AdamW for embedding/lm_head, Muon for matrix params)
# Scale LR by 1/√(model_dim/768) for embeddings/lm_head
dmodel_lr_scale = (model_dim / 768) ** -0.5
print0(f"Scaling the LR for AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

# Separate out parameters into 3 groups (matrix, embedding, lm_head)
# Collect parameter names for each group
# In MLX, we need to use tree_flatten_with_keys from mlx.utils
def get_param_groups(model):
    """Separate model parameters into groups for different optimizers."""
    from mlx.utils import tree_flatten

    embedding_names = []
    lm_head_names = []
    matrix_names = []

    # Get all parameters with their paths
    params_dict = model.parameters()

    def flatten_with_path(tree, prefix=""):
        """Flatten nested dict/list into list of (path,) tuples."""
        result = []
        if isinstance(tree, dict):
            for key, value in tree.items():
                new_prefix = f"{prefix}.{key}" if prefix else key
                if isinstance(value, (dict, list)):
                    result.extend(flatten_with_path(value, new_prefix))
                else:
                    # This is a leaf (actual parameter)
                    result.append(new_prefix)
        elif isinstance(tree, list):
            # Handle lists (like transformer.h which is a list of layers)
            for idx, value in enumerate(tree):
                new_prefix = f"{prefix}.{idx}" if prefix else str(idx)
                if isinstance(value, (dict, list)):
                    result.extend(flatten_with_path(value, new_prefix))
                else:
                    result.append(new_prefix)
        return result

    param_names = flatten_with_path(params_dict)

    for name in param_names:
        if "transformer.wte" in name:
            embedding_names.append(name)
        elif "lm_head" in name:
            lm_head_names.append(name)
        else:
            # All other params (attention Q/K/V/proj, MLP fc/proj) are matrix params
            matrix_names.append(name)

    return embedding_names, lm_head_names, matrix_names

embedding_param_names, lm_head_param_names, matrix_param_names = get_param_groups(model)

# Debug: print sample parameter names to see the structure
total_params = len(embedding_param_names) + len(lm_head_param_names) + len(matrix_param_names)
if total_params < 50:
    print0("DEBUG: Sample parameter names:")
    all_names = embedding_param_names + lm_head_param_names + matrix_param_names
    for name in all_names[:10]:  # Show first 10
        print0(f"  {name}")
    if len(all_names) > 10:
        print0(f"  ... and {len(all_names) - 10} more")

print0(f"Parameter groups:")
print0(f"  Embedding params: {len(embedding_param_names)}")
print0(f"  LM head params: {len(lm_head_param_names)}")
print0(f"  Matrix params (Muon): {len(matrix_param_names)}")

# Create separate optimizers for different parameter groups
# AdamW for embedding (with embedding_lr)
adamw_embedding_lr = embedding_lr * dmodel_lr_scale
adamw_embedding = optim.AdamW(
    learning_rate=adamw_embedding_lr,
    betas=(0.8, 0.95),
    eps=1e-10,
    weight_decay=weight_decay
)

# AdamW for lm_head (with unembedding_lr)
adamw_lm_head_lr = unembedding_lr * dmodel_lr_scale
adamw_lm_head = optim.AdamW(
    learning_rate=adamw_lm_head_lr,
    betas=(0.8, 0.95),
    eps=1e-10,
    weight_decay=weight_decay
)

# Muon for matrix parameters (attention and MLP weights)
muon_optimizer = Muon(
    learning_rate=matrix_lr,
    momentum=0.95,
    nesterov=True,
    backend='newtonschulz5',
    backend_steps=5
)

# Store initial learning rates for scheduling
initial_lrs = {
    "embedding": adamw_embedding_lr,
    "lm_head": adamw_lm_head_lr,
    "matrix": matrix_lr,
}

# Package optimizers and their parameter groups
param_groups = [
    {"name": "embedding", "optimizer": adamw_embedding, "param_names": embedding_param_names, "initial_lr": adamw_embedding_lr},
    {"name": "lm_head", "optimizer": adamw_lm_head, "param_names": lm_head_param_names, "initial_lr": adamw_lm_head_lr},
    {"name": "matrix", "optimizer": muon_optimizer, "param_names": matrix_param_names, "initial_lr": matrix_lr},
]

# -----------------------------------------------------------------------------
# Helper function for multi-optimizer training (used in both compiled and uncompiled paths)
def extract_group_grads(all_grads, param_names, model_params):
    """Extract gradients for specific parameters by navigating the nested structure."""
    def navigate_and_extract(grad_tree, param_tree, current_path=""):
        """Recursively navigate gradient and parameter trees to extract matching grads."""
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
            # Leaf node - check if this parameter should be in the group
            if current_path in param_names:
                return grad_tree
            return None

    return navigate_and_extract(all_grads, model_params)

# -----------------------------------------------------------------------------
# FULL-GRAPH COMPILATION: Compile forward + backward + optimizer updates together
# This matches PyTorch's torch.compile() which optimizes the entire training step
print0("\nSetting up full-graph compilation (forward + backward + optimizer updates)...")

# Capture model state and all optimizer states for compilation
optimizer_states = [group["optimizer"].state for group in param_groups]
full_state_for_compile = [model.state] + optimizer_states

print0(f"  Capturing state: model.state + {len(optimizer_states)} optimizer states")

@partial(mx.compile, inputs=full_state_for_compile, outputs=full_state_for_compile)
def _compiled_training_step(inputs, targets):
    """Fully compiled training step: forward + backward + optimizer updates.

    This compiles the entire training step including gradient computation,
    clipping, and multi-optimizer updates - matching PyTorch's torch.compile().
    """

    # 1. Forward + Backward
    loss_and_grad = nn.value_and_grad(model, compute_loss)
    loss, grads = loss_and_grad(model, inputs, targets)

    # 2. Gradient clipping
    if grad_clip > 0.0:
        from mlx.utils import tree_flatten
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

    # 3. Multi-optimizer updates
    model_params = model.parameters()
    for group in param_groups:
        group_grads = extract_group_grads(grads, group["param_names"], model_params)
        if group_grads is not None:
            group["optimizer"].update(model, group_grads)

    return loss

print0("  Full-graph compilation function created")
print0("  This compiles: forward + backward + grad_clip + multi-optimizer updates")

# Compiled training step flag
USE_FULL_GRAPH_COMPILE = True  # Set to False to use partial compilation (forward+backward only)

# -----------------------------------------------------------------------------
# Learning rate scheduler
def get_lr_multiplier(it):
    warmup_iters = round(warmup_ratio * num_iterations)
    warmdown_iters = round(warmdown_ratio * num_iterations)
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    elif it <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * final_lr_frac

# -----------------------------------------------------------------------------
# Training loop
min_val_bpb = float("inf")
smooth_train_loss = 0.0
ema_beta = 0.9
total_training_time = 0.0
smooth_step_time = 0.0
start_step = 0

print0("Starting training...")
x, y = next(train_loader)  # Get first batch

# Debug: Snapshot initial parameters to verify they change
# DISABLED by default to avoid duplicating model memory (~1GB+ for bf16, ~2GB+ for fp32)
DEBUG_SNAPSHOT_PARAMS = False

initial_params = None
if DEBUG_SNAPSHOT_PARAMS:
    def snapshot_params(tree, prefix=""):
        """Recursively snapshot all leaf parameters with their full paths."""
        result = {}
        if isinstance(tree, dict):
            for k, v in tree.items():
                new_prefix = f"{prefix}.{k}" if prefix else k
                if isinstance(v, (dict, list)):
                    result.update(snapshot_params(v, new_prefix))
                else:
                    result[new_prefix] = mx.array(v)  # Make a copy
        elif isinstance(tree, list):
            for idx, v in enumerate(tree):
                new_prefix = f"{prefix}.{idx}" if prefix else str(idx)
                if isinstance(v, (dict, list)):
                    result.update(snapshot_params(v, new_prefix))
                else:
                    result[new_prefix] = mx.array(v)
        return result

    initial_params = snapshot_params(model.parameters())
    print0(f"DEBUG: Captured initial parameter snapshot ({len(initial_params)} parameters)")

for step in range(start_step, num_iterations + 1):
    last_step = step == num_iterations
    flops_so_far = num_flops_per_token * total_batch_size * step
    
    # Validation evaluation (skip at step 0 to avoid long wait before training starts)
    # Also skip if eval_every is large (validation disabled)
    if step % eval_every == 0 and step > 0 and eval_every < num_iterations:
        val_loader = build_val_loader()
        eval_steps = eval_tokens // (device_batch_size * max_seq_len)
        val_bpb = evaluate_bpb_mlx(model, val_loader, eval_steps, token_bytes)
        print0(f"Step {step:05d} | Validation bpb: {val_bpb:.4f}")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "val/bpb": val_bpb,
        })
    
    # Sampling (matching PyTorch version)
    if sample_every > 0 and (last_step or (step > 0 and step % sample_every == 0)):
        prompts = [
            "The capital of France is",
            "The chemical symbol of gold is",
            "If yesterday was Friday, then tomorrow will be",
            "The opposite of hot is",
            "The planets of the solar system are:",
            "My favorite color is",
            "If 5*x + 3 = 13, then x is",
        ]
        # Simple sampling function for MLX (with KV cache)
        # Note: MLX models don't have eval() mode like PyTorch, but we should
        # ensure we're not computing gradients during sampling
        def sample_text(prompt, max_tokens=16, temperature=0.0):
            # Simple sampling without KV cache (slower but simpler)
            tokens = tokenizer(prompt, prepend="<|bos|>")
            
            # Autoregressive generation (simple, no KV cache)
            for _ in range(max_tokens):
                # Process full sequence each time (inefficient but simple)
                input_ids = mx.array([tokens])
                logits = model(input_ids)
                next_logits = logits[0, -1, :]  # Get last token logits
                # Single eval after getting logits
                mx.eval(next_logits)

                # Sample next token
                if temperature == 0.0:
                    next_token = int(mx.argmax(next_logits))
                else:
                    next_logits = next_logits / temperature
                    probs = mx.softmax(next_logits)
                    next_token = int(mx.random.categorical(probs))

                tokens.append(next_token)

                # Check for stop tokens
                bos_token = tokenizer.get_bos_token_id()
                assistant_end = tokenizer.encode_special("<|assistant_end|>")
                if next_token == bos_token or next_token == assistant_end:
                    break

            return tokenizer.decode(tokens)
        
        for prompt in prompts:
            try:
                sample = sample_text(prompt, max_tokens=16, temperature=0)
                print0(sample)
            except Exception as e:
                print0(f"Sampling failed for '{prompt}': {e}")
                import traceback
                print0(traceback.format_exc())
    
    if last_step:
        break
    
    # Training step
    t0 = time.time()

    # Learning rate scheduling (must happen BEFORE training step)
    lrm = get_lr_multiplier(step)

    # Update learning rates for each optimizer
    for group in param_groups:
        group["optimizer"].learning_rate = group["initial_lr"] * lrm

    # Muon momentum scheduling (matching PyTorch)
    def get_muon_momentum(it):
        frac = min(it / 300, 1)
        momentum = (1 - frac) * 0.85 + frac * 0.95
        return momentum

    muon_momentum = get_muon_momentum(step)
    # Update Muon momentum
    for group in param_groups:
        if group["name"] == "matrix":
            group["optimizer"].momentum = muon_momentum

    if USE_FULL_GRAPH_COMPILE:
        # FULL-GRAPH COMPILATION PATH: Use fully compiled training step
        # This compiles: forward + backward + grad_clip + multi-optimizer updates
        # Matching PyTorch's torch.compile() full-graph optimization

        if grad_accum_steps == 1:
            # Simple case: no gradient accumulation
            loss = _compiled_training_step(x, y)
            accumulated_loss = float(loss)
            # Evaluation is done inside the compiled function
            mx.eval(model.parameters(), *[group["optimizer"].state for group in param_groups])
            grad_norm_val = 0.0  # Not tracked in full-graph mode
        else:
            # Gradient accumulation case: run multiple micro-steps
            # Note: This still uses partial compilation for now, as accumulation complicates full compilation
            accumulated_loss = 0.0
            accumulated_grads = None

            for micro_step in range(grad_accum_steps):
                loss, grads = loss_and_grad_fn(model, x, y)
                accumulated_loss += float(loss)

                # Accumulate gradients
                if accumulated_grads is None:
                    accumulated_grads = grads
                else:
                    accumulated_grads = tree_map(lambda a, b: a + b, accumulated_grads, grads)

                # Get next batch
                if micro_step < grad_accum_steps - 1:
                    x, y = next(train_loader)

            # Average gradients
            accumulated_grads = tree_map(lambda g: g / grad_accum_steps, accumulated_grads)

            # Gradient clipping + optimizer updates (not compiled in this path)
            if grad_clip > 0.0:
                flat_grads = tree_flatten(accumulated_grads)
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
                    grad_norm_val = float(grad_norm)
                    if grad_norm_val > grad_clip:
                        scale = grad_clip / grad_norm_val
                        accumulated_grads = tree_map(lambda g: g * scale if g is not None else None, accumulated_grads)
                else:
                    grad_norm_val = 0.0
            else:
                grad_norm_val = 0.0

            # Multi-optimizer updates
            optimizer_states_to_eval = []
            for group in param_groups:
                group_grads = extract_group_grads(accumulated_grads, group["param_names"], model.parameters())
                if group_grads is not None:
                    group["optimizer"].update(model, group_grads)
                    optimizer_states_to_eval.append(group["optimizer"].state)

            mx.eval(model.parameters(), *optimizer_states_to_eval)
    else:
        # PARTIAL COMPILATION PATH (original): Only forward+backward compiled
        accumulated_loss = 0.0
        accumulated_grads = None

        # Gradient accumulation
        for micro_step in range(grad_accum_steps):
            loss, grads = loss_and_grad_fn(model, x, y)
            accumulated_loss += float(loss)

            # Accumulate gradients
            if accumulated_grads is None:
                accumulated_grads = grads
            else:
                accumulated_grads = tree_map(lambda a, b: a + b, accumulated_grads, grads)

            # Get next batch
            if micro_step < grad_accum_steps - 1:
                x, y = next(train_loader)

        # Average gradients
        accumulated_grads = tree_map(lambda g: g / grad_accum_steps, accumulated_grads)

        # Gradient clipping
        if grad_clip > 0.0:
            flat_grads = tree_flatten(accumulated_grads)
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
                grad_norm_val = float(grad_norm)
                if grad_norm_val > grad_clip:
                    scale = grad_clip / grad_norm_val
                    accumulated_grads = tree_map(lambda g: g * scale if g is not None else None, accumulated_grads)
            else:
                grad_norm_val = 0.0
        else:
            grad_norm_val = 0.0

        # Multi-optimizer updates
        optimizer_states_to_eval = []
        for group in param_groups:
            group_grads = extract_group_grads(accumulated_grads, group["param_names"], model.parameters())
            if group_grads is not None:
                group["optimizer"].update(model, group_grads)
                optimizer_states_to_eval.append(group["optimizer"].state)

        mx.eval(model.parameters(), *optimizer_states_to_eval)

    # Debug: Check if parameters actually changed after first step
    if step == 1 and DEBUG_SNAPSHOT_PARAMS and initial_params is not None:
        def count_param_changes(current, snapshot, path=""):
            """Recursively count changed parameters."""
            if isinstance(current, dict):
                total = 0
                for k, v in current.items():
                    new_path = f"{path}.{k}" if path else k
                    if k in snapshot:
                        total += count_param_changes(v, snapshot[k], new_path)
                return total
            elif isinstance(current, list):
                total = 0
                for idx, v in enumerate(current):
                    new_path = f"{path}.{idx}" if path else str(idx)
                    if idx < len(snapshot):
                        total += count_param_changes(v, snapshot[idx], new_path)
                return total
            else:
                # Leaf - actual parameter array
                if path in initial_params:
                    diff = float(mx.sum(mx.abs(current - initial_params[path])))
                    return 1 if diff > 1e-6 else 0
                return 0

        # Rebuild initial_params with full paths
        from mlx.utils import tree_flatten
        def get_all_param_paths(tree, prefix=""):
            paths = {}
            if isinstance(tree, dict):
                for k, v in tree.items():
                    new_prefix = f"{prefix}.{k}" if prefix else k
                    if isinstance(v, (dict, list)):
                        paths.update(get_all_param_paths(v, new_prefix))
                    else:
                        paths[new_prefix] = v
            elif isinstance(tree, list):
                for idx, v in enumerate(tree):
                    new_prefix = f"{prefix}.{idx}" if prefix else str(idx)
                    if isinstance(v, (dict, list)):
                        paths.update(get_all_param_paths(v, new_prefix))
                    else:
                        paths[new_prefix] = v
            return paths

        current_params = get_all_param_paths(model.parameters())
        changes = sum(1 for name, param in current_params.items()
                     if name in initial_params and
                     float(mx.sum(mx.abs(param - initial_params[name]))) > 1e-6)
        print0(f"DEBUG: After step 1, {changes}/{len(current_params)} parameters changed")

        # Free the snapshot to avoid memory bloat
        del initial_params
        initial_params = None
        print0("DEBUG: Freed parameter snapshot")
    
    # Get next batch for next iteration
    x, y = next(train_loader)
    
    t1 = time.time()
    dt = t1 - t0
    
    # Logging
    train_loss = accumulated_loss / grad_accum_steps
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))
    pct_done = 100 * step / num_iterations
    tok_per_sec = int(total_batch_size / dt) if dt > 0 else 0
    
    if step > 10:
        total_training_time += dt
        smooth_step_time = ema_beta * smooth_step_time + (1 - ema_beta) * dt
    
    # ETA calculation
    steps_remaining = num_iterations - step
    if step > 10 and smooth_step_time > 0:
        estimated_time_remaining = steps_remaining * smooth_step_time
        if estimated_time_remaining < 60:
            time_remaining_str = f"{estimated_time_remaining:.1f}s"
        elif estimated_time_remaining < 3600:
            time_remaining_str = f"{estimated_time_remaining/60:.1f}m"
        else:
            hours = int(estimated_time_remaining // 3600)
            minutes = int((estimated_time_remaining % 3600) // 60)
            time_remaining_str = f"{hours}h{minutes}m"
    else:
        time_remaining_str = "N/A"
    
    print_grad_norm = f" grad norm: {grad_norm_val:.4f} |" if grad_clip > 0.0 else ""
    print0(f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} |{print_grad_norm} lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | total time: {total_training_time/60:.2f}m | ETA: {time_remaining_str}")

    # Optional memory logging (MLX unified memory stats)
    if MEMLOG_EVERY > 0 and step % MEMLOG_EVERY == 0:
        active_gb = mx.get_active_memory() / (1024**3)
        peak_gb = mx.get_peak_memory() / (1024**3)
        cache_gb = mx.get_cache_memory() / (1024**3)
        print0(f"[mem] active={active_gb:.2f}GB peak={peak_gb:.2f}GB cache={cache_gb:.2f}GB")

    if step % 100 == 0:
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "train/loss": debiased_smooth_loss,
            "train/lrm": lrm,
            "train/dt": dt,
            "train/tok_per_sec": tok_per_sec,
        })
        if grad_clip > 0.0:
            wandb_run.log({"train/grad_norm": grad_norm_val})

    # Checkpoint saving
    if save_every > 0 and step % save_every == 0 and step > 0:
        import os
        from mlx.utils import tree_flatten
        checkpoint_dir = os.path.join(get_base_dir(), "checkpoints_mlx")
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"model_step{step:05d}.safetensors")
        print0(f"Saving checkpoint to {checkpoint_path}")
        # Flatten nested parameter dict to flat dict for safetensors
        flat_params = dict(tree_flatten(model.parameters()))
        mx.save_safetensors(str(checkpoint_path), flat_params)
        print0(f"Checkpoint saved")

# Final checkpoint save
import os
from mlx.utils import tree_flatten
checkpoint_dir = os.path.join(get_base_dir(), "checkpoints_mlx")
os.makedirs(checkpoint_dir, exist_ok=True)
final_checkpoint_path = os.path.join(checkpoint_dir, f"model_final_step{num_iterations:05d}.safetensors")
print0(f"\nSaving final checkpoint to {final_checkpoint_path}")
# Flatten nested parameter dict to flat dict for safetensors
flat_params = dict(tree_flatten(model.parameters()))
mx.save_safetensors(str(final_checkpoint_path), flat_params)
print0(f"Final checkpoint saved")

print0(f"Total training time: {total_training_time/60:.2f}m")
print0(f"Minimum validation bpb: {min_val_bpb:.4f}")

wandb_run.finish()

