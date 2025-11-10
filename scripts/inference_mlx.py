"""
Standalone inference script for MLX nanochat model.

Usage:
    # Interactive mode
    python -m scripts.inference_mlx --checkpoint=model.safetensors

    # Single prompt
    python -m scripts.inference_mlx --checkpoint=model.safetensors --prompt="Hello, world"

    # Multiple prompts from file
    python -m scripts.inference_mlx --checkpoint=model.safetensors --prompts_file=prompts.txt
"""

import argparse
import sys
import mlx.core as mx
import mlx.nn as nn

from nanochat.tokenizer import get_tokenizer
from nanochat.common import print0

# Import MLX model
try:
    from mlx_lm.models.nanochat import Model as MLXNanoChatModel, ModelArgs as MLXModelArgs
    from mlx_lm.models.base import KVCache
except ImportError:
    raise ImportError("Please install MLX dependencies: uv sync --extra mlx")


def load_model(checkpoint_path, depth=20):
    """Load trained MLX model from checkpoint."""
    # Model architecture (must match training)
    vocab_size = 65536
    num_layers = depth
    model_dim = depth * 64  # aspect ratio 64
    num_heads = max(1, (model_dim + 127) // 128)  # head dim 128
    num_kv_heads = num_heads  # 1:1 GQA ratio
    max_seq_len = 2048

    # Create model
    model_args = MLXModelArgs(
        model_type="nanochat",
        vocab_size=vocab_size,
        hidden_size=model_dim,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        intermediate_size=model_dim * 4,
        max_position_embeddings=max_seq_len,
        rope_theta=10000.0,
        rope_traditional=False,
    )

    model = MLXNanoChatModel(model_args)

    # Load weights if checkpoint provided
    if checkpoint_path:
        print0(f"Loading checkpoint from {checkpoint_path}")
        weights = mx.load(checkpoint_path)
        model.load_weights(list(weights.items()))
        print0("Checkpoint loaded successfully")
    else:
        print0("Warning: No checkpoint provided, using random initialization")

    return model, model_args


def sample_text(model, tokenizer, prompt, max_tokens=50, temperature=0.7, top_p=0.9):
    """
    Generate text from a prompt using the MLX model.

    Args:
        model: MLX nanochat model
        tokenizer: Tokenizer instance
        prompt: Input text prompt
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (0.0 = greedy, higher = more random)
        top_p: Nucleus sampling threshold

    Returns:
        Generated text (includes prompt)
    """
    # Get model config
    model_args = model.args if hasattr(model, 'args') else None
    if model_args is None:
        # Infer from model structure
        num_layers = len(model.transformer.h)
        num_heads = model.transformer.h[0].attn.num_heads
        model_dim = model.transformer.h[0].attn.hidden_size
        max_seq_len = 2048
    else:
        num_layers = model_args.num_hidden_layers
        num_heads = model_args.num_attention_heads
        model_dim = model_args.hidden_size
        max_seq_len = model_args.max_position_embeddings

    # Tokenize prompt
    tokens = tokenizer(prompt, prepend="<|bos|>")

    # Initialize KV cache
    head_dim = model_dim // num_heads
    num_kv_heads = model.transformer.h[0].attn.num_key_value_heads if hasattr(model.transformer.h[0].attn, 'num_key_value_heads') else num_heads
    cache = [KVCache(1, max_seq_len, num_heads, head_dim, num_kv_heads)
             for _ in range(num_layers)]

    # Prefill: process the prompt
    input_ids = mx.array([tokens])
    mx.eval()
    logits = model(input_ids, cache=cache)
    mx.eval()
    next_logits = logits[0, -1, :]

    # Sample first token
    next_token = sample_token(next_logits, temperature, top_p)
    tokens.append(next_token)

    # Autoregressive generation
    for _ in range(max_tokens - 1):
        input_ids = mx.array([[next_token]])
        mx.eval()
        logits = model(input_ids, cache=cache)
        mx.eval()
        next_logits = logits[0, -1, :]

        next_token = sample_token(next_logits, temperature, top_p)
        tokens.append(next_token)

        # Check for stop tokens
        bos_token = tokenizer.get_bos_token_id()
        assistant_end = tokenizer.encode_special("<|assistant_end|>")
        if next_token == bos_token or next_token == assistant_end:
            break

    return tokenizer.decode(tokens)


def sample_token(logits, temperature=0.7, top_p=0.9):
    """Sample a token from logits with temperature and top-p sampling."""
    if temperature == 0.0:
        return int(mx.argmax(logits))

    # Apply temperature
    logits = logits / temperature
    probs = mx.softmax(logits)

    # Top-p (nucleus) sampling
    if top_p < 1.0:
        sorted_indices = mx.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        cumsum = mx.cumsum(sorted_probs)

        # Find cutoff index
        cutoff_idx = int(mx.argmax((cumsum > top_p).astype(mx.int32)))
        if cutoff_idx == 0:
            cutoff_idx = 1  # Keep at least one token

        # Zero out probabilities beyond cutoff
        top_indices = sorted_indices[:cutoff_idx]
        filtered_probs = mx.zeros_like(probs)
        filtered_probs[top_indices] = probs[top_indices]

        # Renormalize
        probs = filtered_probs / mx.sum(filtered_probs)

    return int(mx.random.categorical(mx.log(probs)))


def interactive_mode(model, tokenizer):
    """Run interactive inference mode."""
    print0("\n=== Interactive Inference Mode ===")
    print0("Type your prompts (Ctrl+C or 'quit' to exit)")
    print0("Commands:")
    print0("  /temp <value>  - Set temperature (default: 0.7)")
    print0("  /tokens <n>    - Set max tokens (default: 50)")
    print0("  /topp <value>  - Set top_p (default: 0.9)")
    print0("")

    temperature = 0.7
    max_tokens = 50
    top_p = 0.9

    while True:
        try:
            prompt = input("\n> ").strip()

            if not prompt:
                continue

            if prompt.lower() in ["quit", "exit", "q"]:
                break

            # Handle commands
            if prompt.startswith("/temp "):
                try:
                    temperature = float(prompt.split()[1])
                    print0(f"Temperature set to {temperature}")
                    continue
                except (ValueError, IndexError):
                    print0("Usage: /temp <value>")
                    continue

            if prompt.startswith("/tokens "):
                try:
                    max_tokens = int(prompt.split()[1])
                    print0(f"Max tokens set to {max_tokens}")
                    continue
                except (ValueError, IndexError):
                    print0("Usage: /tokens <n>")
                    continue

            if prompt.startswith("/topp "):
                try:
                    top_p = float(prompt.split()[1])
                    print0(f"Top-p set to {top_p}")
                    continue
                except (ValueError, IndexError):
                    print0("Usage: /topp <value>")
                    continue

            # Generate
            output = sample_text(model, tokenizer, prompt, max_tokens, temperature, top_p)
            print0(f"\n{output}\n")

        except KeyboardInterrupt:
            print0("\nExiting...")
            break
        except Exception as e:
            print0(f"Error: {e}")
            import traceback
            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="MLX nanochat inference")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint (.safetensors)")
    parser.add_argument("--depth", type=int, default=20, help="Model depth (must match training)")
    parser.add_argument("--prompt", type=str, default=None, help="Single prompt for inference")
    parser.add_argument("--prompts_file", type=str, default=None, help="File with prompts (one per line)")
    parser.add_argument("--max_tokens", type=int, default=50, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p (nucleus) sampling threshold")

    args = parser.parse_args()

    # Load tokenizer
    print0("Loading tokenizer...")
    tokenizer = get_tokenizer()

    # Load model
    print0(f"Loading model (depth={args.depth})...")
    model, model_args = load_model(args.checkpoint, args.depth)

    # Single prompt mode
    if args.prompt:
        print0(f"\nPrompt: {args.prompt}")
        output = sample_text(model, tokenizer, args.prompt, args.max_tokens, args.temperature, args.top_p)
        print0(f"Output: {output}\n")

    # Multi-prompt file mode
    elif args.prompts_file:
        with open(args.prompts_file, 'r') as f:
            prompts = [line.strip() for line in f if line.strip()]

        for i, prompt in enumerate(prompts, 1):
            print0(f"\n[{i}/{len(prompts)}] Prompt: {prompt}")
            output = sample_text(model, tokenizer, prompt, args.max_tokens, args.temperature, args.top_p)
            print0(f"Output: {output}")

    # Interactive mode
    else:
        interactive_mode(model, tokenizer)


if __name__ == "__main__":
    main()
