# MLX Implementation Quick Reference

## Quick Start

```bash
# Install dependencies
uv sync --extra mlx

# Run training
python -m scripts.base_train_mlx \
  --depth=4 \
  --max_seq_len=1024 \
  --device_batch_size=1 \
  --total_batch_size=1024 \
  --num_iterations=100 \
  --run=mlx-test
```

## Key Differences from PyTorch

| Feature | PyTorch | MLX | Status |
|---------|---------|-----|--------|
| Optimizers | AdamW + Muon (separate) | Single AdamW | ⚠️ Needs fix |
| RoPE | Built-in | Manual implementation | ✅ Working |
| Weight Init | `init_weights()` | `init_weights_pytorch_style()` | ✅ Implemented |
| Loss | `F.cross_entropy` | Manual log_softmax | ✅ Working |
| Validation | `evaluate_bpb()` | Placeholder (0.0) | ❌ Missing |
| Sampling | `Engine.generate_batch()` | Custom function | ⚠️ Works but gibberish |

## Critical Issues

1. **Loss too high**: ~17 vs PyTorch ~11 (initial), ~7.5 (after 100 steps)
2. **Single optimizer**: Should use separate optimizers like PyTorch
3. **No Muon**: Matrix layers should use Muon optimizer

## File Structure

```
scripts/base_train_mlx.py  # Main MLX training script
├── apply_rotary_emb_grad_safe()  # Manual RoPE (gradient-safe)
├── AttentionGradSafe  # Custom attention with manual RoPE
├── create_grad_safe_model()  # Model factory
├── init_weights_pytorch_style()  # Weight initialization
├── compute_loss()  # Loss computation
├── loss_and_grad_fn()  # Gradient computation
└── Training loop  # Main training logic
```

## Debugging Checklist

- [ ] Check if gradients are non-zero: `grad_norm > 0`
- [ ] Verify optimizer is updating: Check parameter changes
- [ ] Compare loss on same batch: MLX vs PyTorch
- [ ] Check weight initialization: Should match PyTorch
- [ ] Verify softcap: Applied once in model forward, not in loss

## Next Steps

1. Implement separate optimizers (AdamW for embedding/lm_head, Muon for matrices)
2. Debug why loss isn't decreasing
3. Add validation bpb calculation
4. Fix sampling (should work once model trains)

## Useful Commands

```bash
# Compare loss on same data
python -m scripts.base_train --depth=4 --num_iterations=1  # PyTorch
python -m scripts.base_train_mlx --depth=4 --num_iterations=1  # MLX

# Check gradients
# Add print statements in loss_and_grad_fn() to inspect gradients
```

