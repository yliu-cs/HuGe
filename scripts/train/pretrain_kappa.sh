#!/usr/bin/env bash
# Kappa Training Launcher
# Auto-detects GPU count for single-node; auto-disables DeepSpeed on single GPU.
# Usage:
#   ./scripts/train/pretrain_kappa.sh                                              # auto-detect GPUs
#   ./scripts/train/pretrain_kappa.sh --num_gpus 4                                 # 4 GPUs, DeepSpeed ZeRO-2
#   ./scripts/train/pretrain_kappa.sh --num_gpus 8 --num_nodes 2 --master_addr X   # multi-node
#   ./scripts/train/pretrain_kappa.sh --hostfile hostfile.txt                      # multi-node (hostfile)
#   ./scripts/train/pretrain_kappa.sh --no_deepspeed                               # force plain DDP
#   ./scripts/train/pretrain_kappa.sh --deepspeed                                  # force DeepSpeed (even single GPU)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON="${PROJECT_ROOT}/.venv/bin/python"
DEEPSPEED="${PROJECT_ROOT}/.venv/bin/deepspeed"
TRAIN_SCRIPT="${PROJECT_ROOT}/scripts/train.py"
CONFIG_NAME="pretrain_kappa_config"

NUM_GPUS=""
NUM_NODES=1
MASTER_ADDR="localhost"
MASTER_PORT="${MASTER_PORT:-29500}"
ZERO_STAGE=2
NO_DEEPSPEED=""
DS_CONFIG=""
HOSTFILE=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --num_gpus)       NUM_GPUS="$2"; shift 2 ;;
        --num_nodes)      NUM_NODES="$2"; shift 2 ;;
        --node_rank)      NODE_RANK="$2"; shift 2 ;;
        --master_addr)    MASTER_ADDR="$2"; shift 2 ;;
        --master_port)    MASTER_PORT="$2"; shift 2 ;;
        --zero_stage)     ZERO_STAGE="$2"; shift 2 ;;
        --ds_config)      DS_CONFIG="$2"; shift 2 ;;
        --config)         CONFIG_NAME="$2"; shift 2 ;;
        --hostfile)       HOSTFILE="$2"; shift 2 ;;
        --no_deepspeed)   NO_DEEPSPEED=true; shift ;;
        --deepspeed)      NO_DEEPSPEED=false; shift ;;
        *)                EXTRA_ARGS+=("$1"); shift ;;
    esac
done

NODE_RANK="${NODE_RANK:-0}"

# Auto-detect GPU count for single-node when not specified
if [ -z "$NUM_GPUS" ] && [ "$NUM_NODES" -eq 1 ] && [ -z "$HOSTFILE" ]; then
    if command -v nvidia-smi &>/dev/null; then
        NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
    fi
    NUM_GPUS="${NUM_GPUS:-1}"
    echo "==> Auto-detected ${NUM_GPUS} GPU(s)"
fi
NUM_GPUS="${NUM_GPUS:-1}"

# Single GPU: auto-disable DeepSpeed (unless explicitly requested)
if [ -z "$NO_DEEPSPEED" ]; then
    if [ "$NUM_GPUS" -eq 1 ] && [ "$NUM_NODES" -eq 1 ] && [ -z "$HOSTFILE" ]; then
        NO_DEEPSPEED=true
        echo "==> Single GPU detected, auto-disabling DeepSpeed (use --deepspeed to override)"
    else
        NO_DEEPSPEED=false
    fi
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_DEBUG="${NCCL_DEBUG:-ERROR}"

if [ "$NO_DEEPSPEED" = true ]; then
    echo "==> Launching with torchrun (DDP, no DeepSpeed)"
    exec "$PYTHON" -m torch.distributed.run \
        --nproc_per_node="$NUM_GPUS" \
        --nnodes="$NUM_NODES" \
        --node_rank="$NODE_RANK" \
        --master_addr="$MASTER_ADDR" \
        --master_port="$MASTER_PORT" \
        "$TRAIN_SCRIPT" "$CONFIG_NAME" \
        --train.no-use-deepspeed \
        "${EXTRA_ARGS[@]}"
else
    DS_ARGS=(
        --num_gpus="$NUM_GPUS"
        --master_port="$MASTER_PORT"
    )

    if [ -n "$HOSTFILE" ]; then
        NUM_NODES=$(grep -cve '^\s*$' "$HOSTFILE")
        DS_ARGS+=(--hostfile="$HOSTFILE" --num_nodes="$NUM_NODES")
        echo "==> Launching with DeepSpeed via hostfile (ZeRO-${ZERO_STAGE}, ${NUM_NODES} nodes x ${NUM_GPUS} GPUs)"
    else
        DS_ARGS+=(
            --num_nodes="$NUM_NODES"
            --master_addr="$MASTER_ADDR"
            --node_rank="$NODE_RANK"
        )
        echo "==> Launching with DeepSpeed (ZeRO-${ZERO_STAGE}, ${NUM_NODES} nodes x ${NUM_GPUS} GPUs)"
    fi

    TRAIN_ARGS=(
        "$CONFIG_NAME"
        --train.use-deepspeed
        --train.zero-stage "$ZERO_STAGE"
    )

    if [ -n "$DS_CONFIG" ]; then
        TRAIN_ARGS+=(--train.deepspeed-config "$DS_CONFIG")
    fi

    TRAIN_ARGS+=("${EXTRA_ARGS[@]}")

    exec "$DEEPSPEED" \
        "${DS_ARGS[@]}" \
        "$TRAIN_SCRIPT" \
        "${TRAIN_ARGS[@]}"
fi
