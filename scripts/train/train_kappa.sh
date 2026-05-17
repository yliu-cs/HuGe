#!/usr/bin/env bash
# Kappa training launcher.
#
# Usage:
#   ./scripts/train/train_kappa.sh <config_name> [flags...]
#
# Where <config_name> is any module under src/config/train/ (without ".py"),
# e.g.:
#
#   ./scripts/train/train_kappa.sh pretrain_egodex_kappa_config
#   ./scripts/train/train_kappa.sh posttrain_he_wbt_kappa_config --num_gpus 4
#   ./scripts/train/train_kappa.sh finetune_prop_ep_kappa_config \
#       --train.pretrained-weights .runs/kappa/<posttrain-run>
#
# Distributed flags (all optional — autodetected for single-node):
#   --num_gpus N               # default: nvidia-smi count, 1 if unavailable
#   --num_nodes N              # default: $PET_NNODES or 1
#   --node_rank N              # default: $PET_NODE_RANK or 0
#   --master_addr HOST         # default: $MASTER_ADDR or localhost
#   --master_port PORT         # default: $MASTER_PORT or 29500
#   --zero_stage N             # default: 2
#   --ds_config PATH           # explicit DeepSpeed JSON
#   --hostfile PATH            # multi-node via deepspeed launcher
#   --no_deepspeed             # force plain DDP (default on 1 GPU single-node)
#   --deepspeed                # force DeepSpeed (override single-GPU auto-off)
#
# Anything else is forwarded verbatim to scripts/train.py — typically
# `--train.* / --data.* / --model.*` config overrides.
#
# Multi-node submission: cluster sets PET_NNODES / PET_NODE_RANK / MASTER_ADDR
# / MASTER_PORT per node. Each node runs this script with no extra flags; the
# IB/NCCL tuning block below auto-applies whenever NUM_NODES > 1.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON="${PROJECT_ROOT}/.venv/bin/python"
DEEPSPEED="${PROJECT_ROOT}/.venv/bin/deepspeed"
TORCHRUN="${PROJECT_ROOT}/.venv/bin/torchrun"
TRAIN_SCRIPT="${PROJECT_ROOT}/scripts/train.py"

if [[ $# -lt 1 || "$1" == -* ]]; then
    echo "Error: missing <config_name> as first positional argument" >&2
    echo "Usage: $0 <config_name> [flags...]" >&2
    exit 1
fi
CONFIG_NAME="$1"; shift

NUM_GPUS=""
NUM_NODES="${PET_NNODES:-1}"
NODE_RANK="${PET_NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29500}"
ZERO_STAGE=2
DS_CONFIG=""
HOSTFILE=""
NO_DEEPSPEED=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --num_gpus)     NUM_GPUS="$2"; shift 2 ;;
        --num_nodes)    NUM_NODES="$2"; shift 2 ;;
        --node_rank)    NODE_RANK="$2"; shift 2 ;;
        --master_addr)  MASTER_ADDR="$2"; shift 2 ;;
        --master_port)  MASTER_PORT="$2"; shift 2 ;;
        --zero_stage)   ZERO_STAGE="$2"; shift 2 ;;
        --ds_config)    DS_CONFIG="$2"; shift 2 ;;
        --hostfile)     HOSTFILE="$2"; shift 2 ;;
        --no_deepspeed) NO_DEEPSPEED=true; shift ;;
        --deepspeed)    NO_DEEPSPEED=false; shift ;;
        *)              EXTRA_ARGS+=("$1"); shift ;;
    esac
done

# Autodetect GPU count on single-node when not specified.
if [ -z "$NUM_GPUS" ] && [ "$NUM_NODES" -eq 1 ] && [ -z "$HOSTFILE" ]; then
    if command -v nvidia-smi &>/dev/null; then
        NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
    fi
    NUM_GPUS="${NUM_GPUS:-1}"
    echo "==> Auto-detected ${NUM_GPUS} GPU(s)"
fi
NUM_GPUS="${NUM_GPUS:-1}"

# Single GPU: auto-disable DeepSpeed (overrideable via --deepspeed).
if [ -z "$NO_DEEPSPEED" ]; then
    if [ "$NUM_GPUS" -eq 1 ] && [ "$NUM_NODES" -eq 1 ] && [ -z "$HOSTFILE" ]; then
        NO_DEEPSPEED=true
        echo "==> Single GPU detected, auto-disabling DeepSpeed (use --deepspeed to override)"
    else
        NO_DEEPSPEED=false
    fi
fi

echo "==> Config: ${CONFIG_NAME}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_DEBUG="${NCCL_DEBUG:-ERROR}"

# Multi-node IB / NCCL tuning. Each var uses `:-` so a caller can override
# via the environment. Skipped on single-node since the IB path is unused
# inside one box.
if [ "$NUM_NODES" -gt 1 ] || [ -n "$HOSTFILE" ]; then
    export NCCL_IB_QPS_PER_CONNECTION="${NCCL_IB_QPS_PER_CONNECTION:-8}"
    export NCCL_GDR_LEVEL="${NCCL_GDR_LEVEL:-4}"
    export NCCL_IB_PCI_RELAXED_ORDERING="${NCCL_IB_PCI_RELAXED_ORDERING:-1}"
    export NCCL_IB_TC="${NCCL_IB_TC:-186}"
    export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
    export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
    export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-bond0}"
    export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond0}"
    export NCCL_IB_TIMEOUT="${NCCL_IB_TIMEOUT:-22}"
    export NCCL_IB_RETRY_CNT="${NCCL_IB_RETRY_CNT:-7}"
    export NCCL_IB_HCA="${NCCL_IB_HCA:-^=mlx5_3,mlx5_4,mlx5_5,mlx5_bond_0}"
    # Memory locking is required for IB/RDMA registered buffers; without it
    # NCCL silently falls back to TCP and throughput collapses.
    ulimit -l unlimited || true
    echo "==> Multi-node NCCL/IB tuning applied (bond0, mlx5)"
fi

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
fi

TRAIN_ARGS=(
    "$CONFIG_NAME"
    --train.use-deepspeed
    --train.zero-stage "$ZERO_STAGE"
)
[ -n "$DS_CONFIG" ] && TRAIN_ARGS+=(--train.deepspeed-config "$DS_CONFIG")
TRAIN_ARGS+=("${EXTRA_ARGS[@]}")

# Multi-node without hostfile = PET_*-style cluster (each node runs this
# script independently). The `deepspeed` launcher requires SSH from a single
# master, so use torchrun and let DeepSpeed initialize inside the script.
if [ "$NUM_NODES" -gt 1 ] && [ -z "$HOSTFILE" ]; then
    echo "==> Launching with torchrun + DeepSpeed engine (ZeRO-${ZERO_STAGE}, ${NUM_NODES} nodes x ${NUM_GPUS} GPUs, node_rank=${NODE_RANK})"
    exec "$TORCHRUN" \
        --nproc_per_node="$NUM_GPUS" \
        --nnodes="$NUM_NODES" \
        --node_rank="$NODE_RANK" \
        --master_addr="$MASTER_ADDR" \
        --master_port="$MASTER_PORT" \
        "$TRAIN_SCRIPT" "${TRAIN_ARGS[@]}"
fi

DS_ARGS=(--num_gpus="$NUM_GPUS" --master_port="$MASTER_PORT")
if [ -n "$HOSTFILE" ]; then
    NUM_NODES=$(grep -cve '^\s*$' "$HOSTFILE")
    DS_ARGS+=(--hostfile="$HOSTFILE" --num_nodes="$NUM_NODES")
    echo "==> Launching with DeepSpeed via hostfile (ZeRO-${ZERO_STAGE}, ${NUM_NODES} nodes x ${NUM_GPUS} GPUs)"
else
    echo "==> Launching with DeepSpeed (ZeRO-${ZERO_STAGE}, single node x ${NUM_GPUS} GPUs)"
fi

exec "$DEEPSPEED" "${DS_ARGS[@]}" "$TRAIN_SCRIPT" "${TRAIN_ARGS[@]}"
