#!/bin/bash

# This script is configured to train your own GPT-2 grade LLM (pretraining + finetuning)
# It is designed to run on a blank 8XH100 GPU node and takes approximately 3 hours to complete.

# 1) Example launch (simplest):
# bash runs/speedrun.sh
# 2) Example launch in a screen session (because the run takes ~3 hours):
# screen -L -Logfile runs/speedrun.log -S speedrun bash runs/speedrun.sh
# 3) Example launch with wandb logging, but see below for setting up wandb first:
# WANDB_RUN=speedrun screen -L -Logfile runs/speedrun.log -S speedrun bash runs/speedrun.sh

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

# Download the first ~2B characters of pretraining dataset
# each data shard is ~250M chars
# so we download 2e9 / 250e6 = 8 data shards at this point
# each shard is ~100MB of text (compressed), so this is about ~800MB of data on disk
# look at dev/repackage_data_reference.py for details on how this data was prepared
python -m nanochat.dataset -n 8
# Immediately also kick off downloading more shards in the background while tokenizer trains
# Approximately 150 shards are needed for GPT-2 capability pretraining, add 20 for padding.
# The maximum total number of shards available in the entire dataset is 6542.
python -m nanochat.dataset -n 170 &
DATASET_DOWNLOAD_PID=$!
# train the tokenizer with vocab size 2**15 = 32768 on ~2B characters of data
python -m scripts.tok_train
# evaluate the tokenizer (report compression ratio etc.)
python -m scripts.tok_eval

# -----------------------------------------------------------------------------
# Base model (pretraining)
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

# pretrain the model
if [ "$USE_TORCHRUN" = true ]; then
    # Linux GPU configuration (8XH100):
    # - d24 model, slightly undertrained to beat GPT-2
    # - decrease data:params ratio from compute optimal 10.5 to 9.5
    run_cmd scripts.base_train -- --depth=24 --target-param-data-ratio=9.5 --device-batch-size=16 --fp8 --run=$WANDB_RUN
else
    # macOS configuration (CPU/MPS):
    # - depth=4 (~37M parameters) - much smaller for testing
    # - max_seq_len=1024 (reduced from 2048)
    # - device_batch_size=1 (reduced from 32)
    # - total_batch_size=1024 (reduced from 524,288)
    # - num_iterations=50 (fixed, instead of auto-calculated)
    # - eval_tokens=4096 (reduced from default)
    # - core_metric_every=50 (more frequent evaluation)
    echo "Running on macOS - using smaller model configuration"
    python -m scripts.base_train --depth=4 --max_seq_len=1024 --device_batch_size=1 --eval_tokens=4096 --core_metric_every=50 --total_batch_size=1024 --num_iterations=50 --run=$WANDB_RUN
fi
# evaluate the model: CORE metric, BPB on train/val, and draw samples
run_cmd scripts.base_eval -- --device-batch-size=16

# -----------------------------------------------------------------------------
# SFT (teach the model conversation special tokens, tool use, multiple choice)

# download 2.3MB of synthetic identity conversations to impart a personality to nanochat
# see dev/gen_synthetic_data.py for details on how this data was prepared and to get a sense of how you can easily tune it
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

# run SFT and eval the model
if [ "$USE_TORCHRUN" = true ]; then
    run_cmd scripts.chat_sft -- --device-batch-size=16 --run=$WANDB_RUN
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
# Generate the full report by putting together all the sections
# report.md is the output and will be copied to current directory for convenience
python -m nanochat.report generate
