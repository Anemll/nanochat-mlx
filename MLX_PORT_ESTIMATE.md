# MLX Port Estimate for nanochat

## Overview
This document estimates the effort required to port nanochat from PyTorch to MLX, using the existing [MLX-LM nanochat model](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/nanochat.py) as a starting point.

## What Already Exists ✅

1. **Model Architecture** - The MLX-LM repository already has a complete nanochat model implementation
   - Location: `mlx_lm/models/nanochat.py`
   - Includes: Attention, MLP, Transformer blocks, full model
   - Status: ✅ Ready to use

## What Needs to be Ported 🔄

### 1. Custom Optimizers (HIGH COMPLEXITY)

#### 1.1 Muon Optimizer
- **Current**: `nanochat/muon.py` (~188 lines)
- **Complexity**: HIGH
- **MLX Requirements**:
  - Port Newton-Schulz iteration (`zeropower_via_newtonschulz5`)
  - Port momentum buffer management
  - Port aspect-ratio scaling
  - Handle MLX's value/grad system (different from PyTorch's `.grad`)
- **Estimated Effort**: 2-3 days
- **Notes**: MLX uses functional gradients via `mx.value_and_grad()`, requires different approach

#### 1.2 Distributed AdamW
- **Current**: `nanochat/adamw.py` (~77 lines)
- **Complexity**: MEDIUM-HIGH
- **MLX Requirements**:
  - Port ZeRO-2 style sharding
  - Port gradient reduce-scatter/all-gather
  - Adapt to MLX's optimizer API
- **Estimated Effort**: 2-3 days
- **Notes**: MLX doesn't have built-in distributed training, may need custom implementation

**Total Optimizer Effort**: 4-6 days

### 2. Data Loading (MEDIUM COMPLEXITY)

#### 2.1 Tokenizing Data Loader
- **Current**: `nanochat/dataloader.py` (~50 lines)
- **Complexity**: MEDIUM
- **MLX Requirements**:
  - Port parquet iteration (can reuse)
  - Port tokenization (can reuse Rust BPE)
  - Convert PyTorch tensors to MLX arrays
  - Handle MLX's async data loading
- **Estimated Effort**: 1-2 days
- **Notes**: Tokenizer (Rust BPE) can be reused, just need to convert outputs

**Total Data Loading Effort**: 1-2 days

### 3. Training Scripts (MEDIUM COMPLEXITY)

#### 3.1 Base Training (`scripts/base_train.py`)
- **Current**: ~413 lines
- **Complexity**: MEDIUM
- **MLX Requirements**:
  - Port training loop
  - Port gradient accumulation
  - Port learning rate scheduling
  - Port evaluation (bpb calculation)
  - Port checkpointing
  - Port wandb integration (should work as-is)
- **Estimated Effort**: 3-4 days
- **Key Changes**:
  - Replace `model(x, y)` with MLX forward pass
  - Replace `loss.backward()` with `mx.value_and_grad()`
  - Replace optimizer `.step()` with MLX optimizer updates
  - Convert all tensor operations to MLX

#### 3.2 Mid Training (`scripts/mid_train.py`)
- **Current**: ~300 lines
- **Complexity**: MEDIUM
- **Estimated Effort**: 2-3 days

#### 3.3 Chat SFT (`scripts/chat_sft.py`)
- **Current**: ~280 lines
- **Complexity**: MEDIUM
- **Estimated Effort**: 2-3 days

#### 3.4 Chat RL (`scripts/chat_rl.py`)
- **Current**: ~400 lines
- **Complexity**: HIGH (complex reward logic)
- **Estimated Effort**: 3-4 days

**Total Training Scripts Effort**: 10-14 days

### 4. Evaluation & Utilities (LOW-MEDIUM COMPLEXITY)

#### 4.1 Loss Evaluation (`nanochat/loss_eval.py`)
- **Current**: ~50 lines
- **Complexity**: LOW
- **Estimated Effort**: 0.5-1 day

#### 4.2 Base Evaluation (`scripts/base_eval.py`)
- **Current**: ~200 lines
- **Complexity**: MEDIUM
- **Estimated Effort**: 1-2 days

#### 4.3 Chat Evaluation (`scripts/chat_eval.py`)
- **Current**: ~300 lines
- **Complexity**: MEDIUM
- **Estimated Effort**: 1-2 days

#### 4.4 Engine/Inference (`nanochat/engine.py`)
- **Current**: ~200 lines
- **Complexity**: MEDIUM
- **Estimated Effort**: 1-2 days
- **Notes**: MLX-LM may have inference utilities we can reuse

**Total Evaluation Effort**: 3.5-7 days

### 5. Checkpointing (LOW COMPLEXITY)

#### 5.1 Checkpoint Manager (`nanochat/checkpoint_manager.py`)
- **Current**: ~140 lines
- **Complexity**: LOW-MEDIUM
- **MLX Requirements**:
  - Port model state dict saving/loading
  - Port optimizer state saving/loading
  - Convert between PyTorch and MLX formats (if needed)
- **Estimated Effort**: 1-2 days
- **Notes**: MLX uses different serialization, need to adapt

**Total Checkpointing Effort**: 1-2 days

### 6. Supporting Infrastructure (LOW-MEDIUM COMPLEXITY)

#### 6.1 Common Utilities (`nanochat/common.py`)
- **Current**: ~100 lines
- **Complexity**: LOW
- **Estimated Effort**: 0.5-1 day

#### 6.2 Report Generation (`nanochat/report.py`)
- **Current**: ~200 lines
- **Complexity**: LOW
- **Estimated Effort**: 0.5-1 day

#### 6.3 Tokenizer Integration
- **Current**: Uses Rust BPE
- **Complexity**: LOW
- **Estimated Effort**: 0.5 day
- **Notes**: Should work as-is, just need to convert outputs

**Total Infrastructure Effort**: 1.5-2.5 days

### 7. Testing & Validation (ONGOING)

- Unit tests for optimizers
- Integration tests for training loop
- Validation against PyTorch results
- Performance benchmarking
- **Estimated Effort**: 3-5 days (ongoing)

## Total Effort Estimate

| Component | Days | Complexity |
|-----------|------|------------|
| Custom Optimizers | 4-6 | HIGH |
| Data Loading | 1-2 | MEDIUM |
| Training Scripts | 10-14 | MEDIUM-HIGH |
| Evaluation & Utilities | 3.5-7 | LOW-MEDIUM |
| Checkpointing | 1-2 | LOW-MEDIUM |
| Infrastructure | 1.5-2.5 | LOW |
| Testing & Validation | 3-5 | MEDIUM |
| **TOTAL** | **24-38 days** | |

**Realistic Estimate**: 4-6 weeks of focused development

## Key Challenges

### 1. MLX's Functional Gradient System
- PyTorch: `loss.backward()` → `optimizer.step()`
- MLX: `loss_fn = mx.value_and_grad(model_fn)` → manual optimizer updates
- **Impact**: Requires significant refactoring of training loops

### 2. Distributed Training
- PyTorch: Built-in DDP support
- MLX: No built-in distributed training
- **Impact**: May need custom implementation or single-GPU only

### 3. Custom Optimizers
- Muon optimizer uses complex Newton-Schulz iteration
- MLX may not have all the same primitives
- **Impact**: May need to reimplement or find MLX equivalents

### 4. Checkpoint Compatibility
- PyTorch checkpoints won't work directly with MLX
- Need conversion utilities or separate checkpoint format
- **Impact**: Can't easily resume from PyTorch checkpoints

## Potential Benefits

1. **Performance**: MLX is typically 2-5x faster than PyTorch MPS on Apple Silicon
2. **Memory**: MLX may have better memory efficiency
3. **Native**: Better integration with Apple hardware

## Recommendations

### Option 1: Incremental Port (RECOMMENDED)
1. Start with inference only (use MLX-LM's model)
2. Port data loading
3. Port base training script
4. Port optimizers
5. Port remaining scripts

### Option 2: Full Port
- Port everything at once
- Higher risk, but cleaner end result

### Option 3: Hybrid Approach
- Keep PyTorch for training
- Use MLX for inference (faster)
- Minimal effort, some benefit

## Dependencies to Add

```toml
mlx = ">=0.20.0"
mlx-lm = ">=0.5.0"
```

## References

- [MLX-LM nanochat model](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/nanochat.py)
- [MLX Documentation](https://ml-explore.github.io/mlx/)
- [MLX-LM Repository](https://github.com/ml-explore/mlx-lm)

