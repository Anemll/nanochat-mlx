#!/bin/bash

# This script is the "Best ChatGPT clone that $100 can buy",
# It is designed to run in ~4 hours on 8XH100 node at $3/GPU/hour.

# 1) Example launch (simplest):
# bash speedrun.sh
# 2) Example launch in a screen session (because the run takes ~4 hours):
# screen -L -Logfile speedrun.log -S speedrun bash speedrun.sh
# 3) Example launch with wandb logging, but see below for setting up wandb first:
# WANDB_RUN=speedrun screen -L -Logfile speedrun.log -S speedrun bash speedrun.sh

# Default intermediate artifacts directory is in ~/.cache/nanochat
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# -----------------------------------------------------------------------------
# Python venv setup with uv

# install uv (if not already installed)
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
# ensure uv is in PATH (needed if it was just installed)
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
# create a .venv local virtual environment (if it doesn't exist)
[ -d ".venv" ] || uv venv
# Detect OS and set appropriate dependencies
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS - use CPU/MPS
    EXTRA_DEPS="cpu"
    USE_TORCHRUN=false
else
    # Linux - use GPU
    EXTRA_DEPS="gpu"
    USE_TORCHRUN=true
fi

# install the repo dependencies
uv sync --extra $EXTRA_DEPS
# activate venv so that `python` uses the project's venv instead of system python
source .venv/bin/activate

# -----------------------------------------------------------------------------
# wandb setup
# If you wish to use wandb for logging (it's nice!, recommended).
# 1) Make sure to first log in to wandb, e.g. run:
#    `wandb login`
# 2) Set the WANDB_RUN environment variable when running this script, e.g.:
#    `WANDB_RUN=d26 bash speedrun.sh`
if [ -z "$WANDB_RUN" ]; then
    # by default use "dummy" : it's handled as a special case, skips logging to wandb
    WANDB_RUN=dummy
fi

# -----------------------------------------------------------------------------
# During the course of the run, we will be writing markdown reports to the report/
# directory in the base dir. This command clears it out and writes a header section
# with a bunch of system info and a timestamp that marks the start of the run.
python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Tokenizer

# Install Rust / Cargo
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

# Build the rustbpe Tokenizer
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

# Download the first ~2B characters of pretraining dataset
# look at dev/repackage_data_reference.py for details on how this data was prepared
# each data shard is ~250M chars
# so we download 2e9 / 250e6 = 8 data shards at this point
# each shard is ~100MB of text (compressed), so this is about ~800MB of data on disk
python -m nanochat.dataset -n 8
# Immediately also kick off downloading more shards in the background while tokenizer trains
# See comment below for why 240 is the right number here
python -m nanochat.dataset -n 240 &
DATASET_DOWNLOAD_PID=$!
# train the tokenizer with vocab size 2**16 = 65536 on ~2B characters of data
python -m scripts.tok_train --max_chars=2000000000
# evaluate the tokenizer (report compression ratio etc.)
python -m scripts.tok_eval

# -----------------------------------------------------------------------------
# Base model (pretraining)

# The d20 model is 561M parameters.
# Chinchilla says #tokens = 20X #params, so we need 561e6 * 20 = 11.2B tokens.
# Assume our tokenizer is 4.8 chars/token, this is 11.2B * 4.8 ~= 54B chars.
# At 250M chars/shard, this is 54B / 250M ~= 216 shards needed for pretraining.
# Round up to 240 for safety. At ~100MB/shard, this downloads ~24GB of data to disk.
# (The total number of shards available in the entire dataset is 1822.)
echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID

# Number of processes/GPUs to use (only used on Linux)
NPROC_PER_NODE=8

# Helper function to run commands with or without torchrun
run_cmd() {
    local module="$1"
    shift
    if [ "$USE_TORCHRUN" = true ]; then
        torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m "$module" "$@"
    else
        # macOS: run directly without torchrun
        python -m "$module" "$@"
    fi
}

# pretrain the d20 model
if [ "$USE_TORCHRUN" = true ]; then
    # Linux GPU configuration (8XH100):
    # - depth=20 (~561M parameters)
    # - Uses Chinchilla scaling: tokens = 20 × params (~11.2B tokens)
    # - device_batch_size=32 (default)
    # - max_seq_len=2048 (default)
    # - Auto-calculates iterations based on target_param_data_ratio=20
    run_cmd scripts.base_train -- --depth=20 --run=$WANDB_RUN
else
    # macOS configuration (CPU/MPS):
    # - depth=4 (~37M parameters) - much smaller for testing
    # - max_seq_len=1024 (reduced from 2048)
    # - device_batch_size=1 (reduced from 32)
    # - total_batch_size=1024 (reduced from 524,288)
    # - num_iterations=50 (fixed, instead of auto-calculated)
    # - eval_tokens=4096 (reduced from default)
    # - core_metric_every=50 (more frequent evaluation)
    # 
    # Example manual run (equivalent to what this script does):
    # WANDB_RUN=macos-d4-test python -m scripts.base_train \
    #   --depth=4 --max_seq_len=1024 --device_batch_size=1 \
    #   --eval_tokens=4096 --core_metric_every=50 \
    #   --total_batch_size=1024 --num_iterations=50 \
    #   --run=macos-d4-test
    echo "Running on macOS - using smaller model configuration"
    python -m scripts.base_train --depth=4 --max_seq_len=1024 --device_batch_size=1 --eval_tokens=4096 --core_metric_every=50 --total_batch_size=1024 --num_iterations=50 --run=$WANDB_RUN
fi
# evaluate the model on a larger chunk of train/val data and draw some samples
run_cmd scripts.base_loss --device_batch_size=1
# evaluate the model on CORE tasks
run_cmd scripts.base_eval --max-per-task=16

# -----------------------------------------------------------------------------
# Midtraining (teach the model conversation special tokens, tool use, multiple choice)

# download 2.3MB of synthetic identity conversations to impart a personality to nanochat
# see dev/gen_sft_data.py for details on how this data was prepared and to get a sense of how you can easily tune it
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

# run midtraining and eval the model
if [ "$USE_TORCHRUN" = true ]; then
    run_cmd scripts.mid_train -- --run=$WANDB_RUN
    run_cmd scripts.chat_eval -- -i mid
else
    # macOS: use smaller configuration
    python -m scripts.mid_train --max_seq_len=1024 --device_batch_size=1 --eval_tokens=4096 --total_batch_size=1024 --num_iterations=100 --run=$WANDB_RUN
    python -m scripts.chat_eval --source=mid --max-new-tokens=128 --max-problems=20
fi

# -----------------------------------------------------------------------------
# Supervised Finetuning (domain adaptation to each sequence all by itself per row)

# train sft and re-eval right away (should see a small bump)
if [ "$USE_TORCHRUN" = true ]; then
    run_cmd scripts.chat_sft -- --run=$WANDB_RUN
    run_cmd scripts.chat_eval -- -i sft
else
    # macOS: use smaller configuration
    python -m scripts.chat_sft --device_batch_size=1 --target_examples_per_step=4 --num_iterations=100 --eval_steps=4 --eval_metrics_max_problems=16 --run=$WANDB_RUN
    python -m scripts.chat_eval --source=sft --max-new-tokens=128 --max-problems=20
fi

# chat with the model over CLI! Leave out the -p to chat interactively
# python -m scripts.chat_cli -p "Why is the sky blue?"

# even better, chat with your model over a pretty WebUI ChatGPT style
# python -m scripts.chat_web

# -----------------------------------------------------------------------------
# Reinforcement Learning. Optional, and currently only on GSM8K
# (optional)

# run reinforcement learning
# torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_rl -- --run=$WANDB_RUN
# eval the RL model only on GSM8K
# torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i rl -a GSM8K

# -----------------------------------------------------------------------------
# Generate the full report by putting together all the sections
# report.md is the output and will be copied to current directory for convenience
python -m nanochat.report generate
