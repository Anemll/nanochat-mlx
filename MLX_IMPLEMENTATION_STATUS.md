# MLX Implementation Status Summary

## Overview
MLX port of `scripts/base_train.py` for pretraining nanochat models on Apple Silicon (MPS). Uses MLX-LM's existing nanochat model as a base, with custom modifications for gradient-safe RoPE.

## File Location
- **Main script**: `scripts/base_train_mlx.py` (721 lines)
- **Based on**: `scripts/base_train.py` (PyTorch version)

## What Was Implemented ✅

### 1. Model Architecture
- ✅ Custom `GradSafeModel` wrapper around MLX-LM nanochat
- ✅ Custom `AttentionGradSafe` layer with manual RoPE implementation
- ✅ Manual RoPE computation (no `mx.fast.rope`) to support gradients
- ✅ PyTorch-style weight initialization matching `nanochat/gpt.py`

### 2. Training Infrastructure
- ✅ Data loading: `mlx_data_loader()` converts PyTorch data to MLX arrays
- ✅ Loss computation: Manual cross-entropy with numerical stability
- ✅ Gradient computation: `mx.value_and_grad()` for backpropagation
- ✅ Gradient accumulation: Multi-step accumulation before optimizer update
- ✅ Gradient clipping: Implemented
- ✅ Learning rate scheduling: Warmup/warmdown matching PyTorch

### 3. Optimizer Setup
- ✅ Single AdamW optimizer (simplified from PyTorch's dual-optimizer setup)
- ⚠️ **NOT IMPLEMENTED**: Separate optimizers for embedding/lm_head/matrix layers
- ⚠️ **NOT IMPLEMENTED**: Muon optimizer for matrix layers

### 4. Evaluation & Sampling
- ✅ Basic sampling function with KV cache
- ⚠️ **NOT IMPLEMENTED**: Validation bpb calculation (`evaluate_bpb`)
- ⚠️ **NOT IMPLEMENTED**: CORE metric evaluation

### 5. Logging
- ✅ Wandb integration
- ✅ Training metrics (loss, grad_norm, tokens/sec, etc.)

## Current Issues 🔴

### 1. **Loss Not Matching PyTorch**
- **Problem**: MLX loss starts at ~17, PyTorch starts at ~11
- **After 100 steps**: MLX ~17, PyTorch ~7.5
- **Possible causes**:
  - Optimizer setup difference (single AdamW vs separate optimizers)
  - Missing Muon optimizer for matrix layers
  - Potential numerical differences in loss computation
  - Weight initialization might not be fully correct

### 2. **Sampling Produces Gibberish**
- **Problem**: Model generates repeated "E" tokens: `EEEEEEEEEEEEEEEE`
- **Cause**: Model not learning (high loss), so sampling is meaningless
- **Status**: Sampling code structure is correct, but model needs to train properly first

### 3. **Missing Features**
- Validation bpb calculation not implemented (returns 0.0)
- CORE metric evaluation not implemented
- Checkpoint saving/loading not implemented
- Resume from checkpoint not implemented

## Key Technical Details

### RoPE Gradient Issue (SOLVED)
- **Problem**: MLX's `mx.fast.rope` doesn't support gradients through `freqs`/`offset`
- **Solution**: Manual RoPE implementation using `mx.cos`, `mx.sin`, and basic operations
- **Location**: `apply_rotary_emb_grad_safe()` function

### Weight Initialization
- Matches PyTorch's `init_weights()`:
  - Linear layers: `std = 1/√fan_in * min(1, √(fan_out/fan_in))`
  - Embeddings: `std=1.0`
  - `lm_head` and all `c_proj`: zeros
- **Location**: `init_weights_pytorch_style()` function

### Model Structure
```python
GradSafeModel
  └── GradSafeNanoChatModel (transformer)
      ├── wte (Embedding)
      └── h (list of TransformerBlockGradSafe)
          ├── attn (AttentionGradSafe)  # Uses manual RoPE
          └── mlp (MLP)
  └── lm_head (Linear)
```

## Differences from PyTorch Version

### Optimizer Setup
| PyTorch | MLX |
|---------|-----|
| AdamW for embedding/lm_head | Single AdamW for all params |
| Muon for matrix layers | Not implemented |
| Separate LRs per group | Averaged LR: `(embedding_lr + unembedding_lr) / 2` |

### Loss Computation
- PyTorch: `F.cross_entropy()` with `ignore_index=-1`
- MLX: Manual log_softmax + gather (no ignore_index handling)

### Model Initialization
- PyTorch: `model.init_weights()` called explicitly
- MLX: `init_weights_pytorch_style()` called after model creation

## Running the Code

### Installation
```bash
uv sync --extra mlx
# or
pip install mlx mlx-lm mlx-optimizers
```

### Basic Usage
```bash
python -m scripts.base_train_mlx \
  --depth=4 \
  --max_seq_len=1024 \
  --device_batch_size=1 \
  --total_batch_size=1024 \
  --num_iterations=100 \
  --run=mlx-test
```

### Current Performance
- **Speed**: ~2,500-3,000 tokens/sec on MPS
- **Memory**: Similar to PyTorch version
- **Training**: Runs without errors, but loss doesn't decrease properly

## Next Steps Needed 🔧

### High Priority
1. **Fix optimizer setup**
   - Implement separate AdamW optimizers for embedding/lm_head
   - Implement Muon optimizer for matrix layers (or use `mlx-optimizers` Muon)
   - Match PyTorch's learning rate scheme

2. **Debug loss computation**
   - Verify loss matches PyTorch on same batch
   - Check for numerical differences
   - Ensure softcap is applied correctly (only once)

3. **Verify gradients**
   - Add debug prints to check gradient magnitudes
   - Verify all parameters are receiving gradients
   - Check if RoPE gradients are flowing correctly

### Medium Priority
4. **Implement validation**
   - Port `evaluate_bpb()` from `nanochat/loss_eval.py`
   - Test on validation set

5. **Fix sampling**
   - Once model trains properly, sampling should work
   - May need to verify KV cache usage

6. **Add checkpointing**
   - Save/load model weights
   - Save/load optimizer state
   - Resume training functionality

### Low Priority
7. **CORE metric evaluation**
   - Port evaluation code from `scripts/base_eval.py`

8. **Performance optimization**
   - Profile and optimize bottlenecks
   - Compare with PyTorch performance

## Key Files Reference

### Modified/Created
- `scripts/base_train_mlx.py` - Main training script

### Referenced (from PyTorch version)
- `scripts/base_train.py` - Original PyTorch training script
- `nanochat/gpt.py` - PyTorch model definition
- `nanochat/loss_eval.py` - Loss evaluation utilities
- `nanochat/engine.py` - Inference engine (for sampling reference)

### MLX Dependencies
- `mlx.core` - Core MLX operations
- `mlx.nn` - Neural network layers
- `mlx.optimizers` - Optimizers (AdamW)
- `mlx_lm.models.nanochat` - Nanochat model from MLX-LM
- `mlx_lm.models.base` - Base utilities (KVCache, etc.)

## Known Limitations

1. **Single optimizer**: All parameters use same learning rate (averaged)
2. **No Muon**: Matrix layers don't use Muon optimizer
3. **No validation**: Can't measure actual model quality
4. **No checkpointing**: Can't resume training
5. **Early training**: Only tested for 100 steps, loss not converging

## Testing Status

- ✅ Code runs without crashes
- ✅ Gradients compute (no RoPE errors)
- ✅ Training loop completes
- ✅ Wandb logging works
- ❌ Loss doesn't decrease properly
- ❌ Sampling produces gibberish
- ❌ Validation not implemented

## Questions for Next Agent

1. Should we implement Muon optimizer or use `mlx-optimizers` Muon?
2. How to debug why loss isn't decreasing? (gradients? optimizer? loss computation?)
3. Should we match PyTorch optimizer setup exactly, or is single optimizer acceptable?
4. Priority: Fix training first, or add missing features (validation, checkpointing)?

## Contact Points

- Original PyTorch implementation: `scripts/base_train.py`
- MLX-LM nanochat model: `mlx_lm.models.nanochat`
- Weight initialization reference: `nanochat/gpt.py::init_weights()`
- Loss computation reference: `nanochat/gpt.py::forward()` (training mode)

---

**Last Updated**: 2025-11-08
**Status**: Functional but not matching PyTorch results
**Next Priority**: Fix optimizer setup and debug loss computation

