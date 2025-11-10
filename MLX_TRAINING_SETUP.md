# MLX Training Setup Guide

## Quick Start

### 1. Install Dependencies

```bash
# Using uv (recommended)
uv sync --extra mlx

# Or if you prefer pip
pip install mlx mlx-lm mlx-optimizers
```

### 2. Test MLX Installation

```bash
python -c "import mlx.core as mx; print('MLX version:', mx.__version__)"
```

### 3. Run MLX Training (100 steps test)

```bash
python -m scripts.base_train_mlx \
  --depth=4 \
  --max_seq_len=1024 \
  --device_batch_size=1 \
  --total_batch_size=1024 \
  --num_iterations=100 \
  --eval_every=50 \
  --run=mlx-test-100steps
```

## Comparison with PyTorch

To compare results, run PyTorch version with same config:

```bash
python -m scripts.base_train \
  --depth=4 \
  --max_seq_len=1024 \
  --device_batch_size=1 \
  --total_batch_size=1024 \
  --num_iterations=100 \
  --eval_every=50 \
  --run=pytorch-test-100steps
```

## Key Differences

### PyTorch
- Uses `loss.backward()` → `optimizer.step()`
- Model compiled with `torch.compile()`
- Distributed training support

### MLX
- Uses `mx.value_and_grad()` → `optimizer.update()`
- Lazy evaluation (need `mx.eval()`)
- Single GPU only (no distributed)

## Current Status

✅ **Working:**
- Model initialization (MLX-LM nanochat)
- Data loading (converted to MLX arrays)
- Basic training loop with AdamW
- Gradient accumulation
- Gradient clipping
- Learning rate scheduling
- Wandb logging

⚠️ **TODO:**
- Per-parameter-group learning rates (currently using single LR)
- Muon optimizer integration
- Validation bpb calculation
- Checkpointing
- CORE metric evaluation

## Next Steps

1. **Test basic training** - Run 100 steps and compare loss curves
2. **Fix per-parameter LR** - Implement separate optimizers or manual scaling
3. **Add Muon optimizer** - Port Muon from mlx-optimizers
4. **Add validation** - Port evaluate_bpb to MLX
5. **Add checkpointing** - Save/load model state

## Troubleshooting

### Import Errors
If you see `ImportError: mlx_lm`, install:
```bash
uv sync --extra mlx
```

### Memory Issues
Reduce `device_batch_size` or `max_seq_len`:
```bash
--device_batch_size=1 --max_seq_len=512
```

### Loss Mismatch
Check:
1. Random seed (not set yet - TODO)
2. Learning rate scaling
3. Loss function implementation

