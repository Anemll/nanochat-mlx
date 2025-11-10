#!/usr/bin/env python
"""
Comprehensive memory profiling for MLX and PyTorch training.

Breaks down memory usage by component:
- Model parameters
- Optimizer states (AdamW m/v, Muon momentum)
- Gradients
- Activations (estimated)

Usage:
    python memory_profile.py --framework mlx --depth 4
    python memory_profile.py --framework pytorch --depth 4
    python memory_profile.py --both --depth 20
"""

import argparse
import sys


def calculate_memory_breakdown_mlx(depth, vocab_size=65536, aspect_ratio=64, use_bfloat16=True):
    """Calculate detailed memory breakdown for MLX training."""
    import mlx.core as mx

    # Model architecture
    num_layers = depth
    model_dim = depth * aspect_ratio
    num_heads = max(1, (model_dim + 127) // 128)
    num_kv_heads = num_heads
    intermediate_size = model_dim * 4

    # Dtype
    dtype_bytes = 2 if use_bfloat16 else 4
    dtype_name = "bfloat16" if use_bfloat16 else "float32"

    print(f"\n{'='*70}")
    print(f"MLX MEMORY BREAKDOWN (depth={depth}, dtype={dtype_name})")
    print(f"{'='*70}")
    print(f"Model: {num_layers} layers, {model_dim} dim, {num_heads} heads")
    print(f"Dtype: {dtype_name} ({dtype_bytes} bytes per param)")
    print()

    # --- Model Parameters ---
    print("📦 MODEL PARAMETERS")
    print("-" * 70)

    # Embedding
    embedding_params = vocab_size * model_dim
    embedding_mb = (embedding_params * dtype_bytes) / (1024**2)
    print(f"  Embedding (wte):        {embedding_params:>12,} params = {embedding_mb:>8.2f} MB")

    # LM Head
    lm_head_params = vocab_size * model_dim
    lm_head_mb = (lm_head_params * dtype_bytes) / (1024**2)
    print(f"  LM Head:                {lm_head_params:>12,} params = {lm_head_mb:>8.2f} MB")

    # Transformer layers (matrix params)
    attn_qkv_params = 3 * model_dim * model_dim  # Q, K, V projections
    attn_proj_params = model_dim * model_dim      # Output projection
    mlp_fc_params = model_dim * intermediate_size
    mlp_proj_params = intermediate_size * model_dim

    per_layer_params = attn_qkv_params + attn_proj_params + mlp_fc_params + mlp_proj_params
    per_layer_mb = (per_layer_params * dtype_bytes) / (1024**2)

    total_layer_params = per_layer_params * num_layers
    total_layer_mb = per_layer_mb * num_layers

    print(f"  Transformer layers:")
    print(f"    Attention (Q,K,V,proj): {attn_qkv_params + attn_proj_params:>12,} params/layer")
    print(f"    MLP (fc,proj):          {mlp_fc_params + mlp_proj_params:>12,} params/layer")
    print(f"    Per-layer total:        {per_layer_params:>12,} params = {per_layer_mb:>8.2f} MB")
    print(f"    All {num_layers} layers:         {total_layer_params:>12,} params = {total_layer_mb:>8.2f} MB")

    total_params = embedding_params + lm_head_params + total_layer_params
    total_params_mb = (total_params * dtype_bytes) / (1024**2)
    total_params_gb = total_params_mb / 1024

    print(f"  {'TOTAL PARAMETERS:':<24} {total_params:>12,} params = {total_params_mb:>8.2f} MB ({total_params_gb:.2f} GB)")
    print()

    # --- Optimizer States ---
    print("🔧 OPTIMIZER STATES")
    print("-" * 70)

    # AdamW for embedding (m, v states)
    adamw_emb_states = embedding_params * 2  # m and v
    adamw_emb_mb = (adamw_emb_states * dtype_bytes) / (1024**2)
    print(f"  AdamW (embedding):      {adamw_emb_states:>12,} states = {adamw_emb_mb:>8.2f} MB (m,v)")

    # AdamW for lm_head (m, v states)
    adamw_lm_states = lm_head_params * 2
    adamw_lm_mb = (adamw_lm_states * dtype_bytes) / (1024**2)
    print(f"  AdamW (lm_head):        {adamw_lm_states:>12,} states = {adamw_lm_mb:>8.2f} MB (m,v)")

    # Muon for matrix params (momentum state only)
    muon_params = total_layer_params
    muon_states = muon_params * 1  # momentum only
    muon_mb = (muon_states * dtype_bytes) / (1024**2)
    print(f"  Muon (matrix params):   {muon_states:>12,} states = {muon_mb:>8.2f} MB (momentum)")

    total_opt_states = adamw_emb_states + adamw_lm_states + muon_states
    total_opt_mb = (total_opt_states * dtype_bytes) / (1024**2)
    total_opt_gb = total_opt_mb / 1024

    print(f"  {'TOTAL OPT STATES:':<24} {total_opt_states:>12,} states = {total_opt_mb:>8.2f} MB ({total_opt_gb:.2f} GB)")
    print()

    # --- Gradients ---
    print("∇ GRADIENTS")
    print("-" * 70)

    grad_params = total_params  # One gradient per parameter
    grad_mb = (grad_params * dtype_bytes) / (1024**2)
    grad_gb = grad_mb / 1024

    print(f"  Gradients:              {grad_params:>12,} grads  = {grad_mb:>8.2f} MB ({grad_gb:.2f} GB)")
    print()

    # --- Activations (rough estimate) ---
    print("📊 ACTIVATIONS (estimated, batch_size=1, seq_len=1024)")
    print("-" * 70)

    batch_size = 1
    seq_len = 1024

    # Per-layer activations: attention outputs + MLP outputs
    attn_activation = batch_size * seq_len * model_dim
    mlp_activation = batch_size * seq_len * intermediate_size
    per_layer_activation = attn_activation + mlp_activation
    total_activation = per_layer_activation * num_layers
    activation_mb = (total_activation * dtype_bytes) / (1024**2)
    activation_gb = activation_mb / 1024

    print(f"  Attention outputs:      ~{attn_activation * num_layers:>12,} elems")
    print(f"  MLP intermediate:       ~{mlp_activation * num_layers:>12,} elems")
    print(f"  Total activations:      ~{total_activation:>12,} elems  = {activation_mb:>8.2f} MB ({activation_gb:.2f} GB)")
    print(f"  (Note: Actual may be higher due to intermediate buffers)")
    print()

    # --- Total ---
    print("💾 TOTAL ESTIMATED MEMORY")
    print("-" * 70)

    total_mb = total_params_mb + total_opt_mb + grad_mb + activation_mb
    total_gb = total_mb / 1024

    print(f"  Parameters:             {total_params_mb:>8.2f} MB ({total_params_gb:.2f} GB)")
    print(f"  Optimizer states:       {total_opt_mb:>8.2f} MB ({total_opt_gb:.2f} GB)")
    print(f"  Gradients:              {grad_mb:>8.2f} MB ({grad_gb:.2f} GB)")
    print(f"  Activations (est):      {activation_mb:>8.2f} MB ({activation_gb:.2f} GB)")
    print(f"  {'─'*70}")
    print(f"  TOTAL:                  {total_mb:>8.2f} MB ({total_gb:.2f} GB)")
    print()

    # Breakdown by percentage
    print("📊 BREAKDOWN BY PERCENTAGE")
    print("-" * 70)
    print(f"  Parameters:             {100*total_params_mb/total_mb:>6.2f}%")
    print(f"  Optimizer states:       {100*total_opt_mb/total_mb:>6.2f}%")
    print(f"  Gradients:              {100*grad_mb/total_mb:>6.2f}%")
    print(f"  Activations:            {100*activation_mb/total_mb:>6.2f}%")
    print()

    return {
        "params_gb": total_params_gb,
        "opt_states_gb": total_opt_gb,
        "gradients_gb": grad_gb,
        "activations_gb": activation_gb,
        "total_gb": total_gb
    }


def calculate_memory_breakdown_pytorch(depth, vocab_size=65536, aspect_ratio=64, use_bfloat16=True):
    """Calculate detailed memory breakdown for PyTorch training."""

    # Model architecture
    num_layers = depth
    model_dim = depth * aspect_ratio
    num_heads = max(1, (model_dim + 127) // 128)
    num_kv_heads = num_heads
    intermediate_size = model_dim * 4

    # Dtype
    dtype_bytes = 2 if use_bfloat16 else 4
    dtype_name = "bfloat16" if use_bfloat16 else "float32"

    print(f"\n{'='*70}")
    print(f"PYTORCH MEMORY BREAKDOWN (depth={depth}, dtype={dtype_name})")
    print(f"{'='*70}")
    print(f"Model: {num_layers} layers, {model_dim} dim, {num_heads} heads")
    print(f"Dtype: {dtype_name} ({dtype_bytes} bytes per param)")
    print()

    # Same parameter counts as MLX
    embedding_params = vocab_size * model_dim
    lm_head_params = vocab_size * model_dim

    attn_qkv_params = 3 * model_dim * model_dim
    attn_proj_params = model_dim * model_dim
    mlp_fc_params = model_dim * intermediate_size
    mlp_proj_params = intermediate_size * model_dim

    per_layer_params = attn_qkv_params + attn_proj_params + mlp_fc_params + mlp_proj_params
    total_layer_params = per_layer_params * num_layers

    total_params = embedding_params + lm_head_params + total_layer_params
    total_params_mb = (total_params * dtype_bytes) / (1024**2)
    total_params_gb = total_params_mb / 1024

    print("📦 MODEL PARAMETERS")
    print("-" * 70)
    print(f"  Embedding (wte):        {embedding_params:>12,} params = {(embedding_params * dtype_bytes) / (1024**2):>8.2f} MB")
    print(f"  LM Head:                {lm_head_params:>12,} params = {(lm_head_params * dtype_bytes) / (1024**2):>8.2f} MB")
    print(f"  Transformer layers:     {total_layer_params:>12,} params = {(total_layer_params * dtype_bytes) / (1024**2):>8.2f} MB")
    print(f"  {'TOTAL PARAMETERS:':<24} {total_params:>12,} params = {total_params_mb:>8.2f} MB ({total_params_gb:.2f} GB)")
    print()

    # PyTorch uses same optimizer structure
    print("🔧 OPTIMIZER STATES")
    print("-" * 70)

    adamw_emb_states = embedding_params * 2
    adamw_emb_mb = (adamw_emb_states * dtype_bytes) / (1024**2)
    print(f"  AdamW (embedding):      {adamw_emb_states:>12,} states = {adamw_emb_mb:>8.2f} MB (m,v)")

    adamw_lm_states = lm_head_params * 2
    adamw_lm_mb = (adamw_lm_states * dtype_bytes) / (1024**2)
    print(f"  AdamW (lm_head):        {adamw_lm_states:>12,} states = {adamw_lm_mb:>8.2f} MB (m,v)")

    muon_states = total_layer_params * 1
    muon_mb = (muon_states * dtype_bytes) / (1024**2)
    print(f"  Muon (matrix params):   {muon_states:>12,} states = {muon_mb:>8.2f} MB (momentum)")

    total_opt_states = adamw_emb_states + adamw_lm_states + muon_states
    total_opt_mb = (total_opt_states * dtype_bytes) / (1024**2)
    total_opt_gb = total_opt_mb / 1024

    print(f"  {'TOTAL OPT STATES:':<24} {total_opt_states:>12,} states = {total_opt_mb:>8.2f} MB ({total_opt_gb:.2f} GB)")
    print()

    # Gradients
    grad_params = total_params
    grad_mb = (grad_params * dtype_bytes) / (1024**2)
    grad_gb = grad_mb / 1024

    print("∇ GRADIENTS")
    print("-" * 70)
    print(f"  Gradients:              {grad_params:>12,} grads  = {grad_mb:>8.2f} MB ({grad_gb:.2f} GB)")
    print()

    # Activations
    batch_size = 1
    seq_len = 1024

    attn_activation = batch_size * seq_len * model_dim
    mlp_activation = batch_size * seq_len * intermediate_size
    per_layer_activation = attn_activation + mlp_activation
    total_activation = per_layer_activation * num_layers
    activation_mb = (total_activation * dtype_bytes) / (1024**2)
    activation_gb = activation_mb / 1024

    print("📊 ACTIVATIONS (estimated, batch_size=1, seq_len=1024)")
    print("-" * 70)
    print(f"  Total activations:      ~{total_activation:>12,} elems  = {activation_mb:>8.2f} MB ({activation_gb:.2f} GB)")
    print()

    # Total
    total_mb = total_params_mb + total_opt_mb + grad_mb + activation_mb
    total_gb = total_mb / 1024

    print("💾 TOTAL ESTIMATED MEMORY")
    print("-" * 70)
    print(f"  Parameters:             {total_params_mb:>8.2f} MB ({total_params_gb:.2f} GB)")
    print(f"  Optimizer states:       {total_opt_mb:>8.2f} MB ({total_opt_gb:.2f} GB)")
    print(f"  Gradients:              {grad_mb:>8.2f} MB ({grad_gb:.2f} GB)")
    print(f"  Activations (est):      {activation_mb:>8.2f} MB ({activation_gb:.2f} GB)")
    print(f"  {'─'*70}")
    print(f"  TOTAL:                  {total_mb:>8.2f} MB ({total_gb:.2f} GB)")
    print()

    print("📊 BREAKDOWN BY PERCENTAGE")
    print("-" * 70)
    print(f"  Parameters:             {100*total_params_mb/total_mb:>6.2f}%")
    print(f"  Optimizer states:       {100*total_opt_mb/total_mb:>6.2f}%")
    print(f"  Gradients:              {100*grad_mb/total_mb:>6.2f}%")
    print(f"  Activations:            {100*activation_mb/total_mb:>6.2f}%")
    print()

    return {
        "params_gb": total_params_gb,
        "opt_states_gb": total_opt_gb,
        "gradients_gb": grad_gb,
        "activations_gb": activation_gb,
        "total_gb": total_gb
    }


def main():
    parser = argparse.ArgumentParser(description="Memory profiling for MLX and PyTorch")
    parser.add_argument("--framework", choices=["mlx", "pytorch", "both"], default="both",
                        help="Which framework to profile")
    parser.add_argument("--depth", type=int, default=4,
                        help="Model depth (number of layers)")
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16",
                        help="Parameter dtype")

    args = parser.parse_args()

    use_bfloat16 = args.dtype == "bfloat16"

    if args.framework in ["mlx", "both"]:
        mlx_stats = calculate_memory_breakdown_mlx(args.depth, use_bfloat16=use_bfloat16)

    if args.framework in ["pytorch", "both"]:
        pytorch_stats = calculate_memory_breakdown_pytorch(args.depth, use_bfloat16=use_bfloat16)

    if args.framework == "both":
        print(f"\n{'='*70}")
        print("COMPARISON: MLX vs PyTorch")
        print(f"{'='*70}")
        print(f"{'Component':<24} {'MLX (GB)':<12} {'PyTorch (GB)':<12} {'Difference'}")
        print("-" * 70)
        print(f"{'Parameters':<24} {mlx_stats['params_gb']:<12.2f} {pytorch_stats['params_gb']:<12.2f} {(mlx_stats['params_gb'] - pytorch_stats['params_gb'])*1024:+.1f} MB")
        print(f"{'Optimizer states':<24} {mlx_stats['opt_states_gb']:<12.2f} {pytorch_stats['opt_states_gb']:<12.2f} {(mlx_stats['opt_states_gb'] - pytorch_stats['opt_states_gb'])*1024:+.1f} MB")
        print(f"{'Gradients':<24} {mlx_stats['gradients_gb']:<12.2f} {pytorch_stats['gradients_gb']:<12.2f} {(mlx_stats['gradients_gb'] - pytorch_stats['gradients_gb'])*1024:+.1f} MB")
        print(f"{'Activations (est)':<24} {mlx_stats['activations_gb']:<12.2f} {pytorch_stats['activations_gb']:<12.2f} {(mlx_stats['activations_gb'] - pytorch_stats['activations_gb'])*1024:+.1f} MB")
        print("-" * 70)
        print(f"{'TOTAL':<24} {mlx_stats['total_gb']:<12.2f} {pytorch_stats['total_gb']:<12.2f} {(mlx_stats['total_gb'] - pytorch_stats['total_gb'])*1024:+.1f} MB")
        print()
        print("Note: Theoretical calculation. Actual memory may differ due to:")
        print("  - Framework overhead, buffers, compilation caches")
        print("  - Activation checkpointing or recomputation")
        print("  - Memory fragmentation")
        print("  - Unified memory (MLX) vs separate CPU/GPU memory (PyTorch)")


if __name__ == "__main__":
    main()
