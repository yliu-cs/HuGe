#!/usr/bin/env bash
# Kappa Training Launcher
# Auto-detects GPU count for single-node; auto-disables DeepSpeed on single GPU.
# Stage selection:
#   --stage egodex                                                                # stage 1 (EgoDex pretrain)
#   --stage he                                                                    # stage 2 (Humanoid-Everyday pretrain)
#   --config <name>                                                               # arbitrary config under src/config/train/
# Exactly one of --stage / --config must be provided.
# Usage:
#   ./scripts/train/pretrain_kappa.sh --stage egodex                              # auto-detect GPUs
#   ./scripts/train/pretrain_kappa.sh --stage he --num_gpus 4                     # 4 GPUs, DeepSpeed ZeRO-2
#   ./scripts/train/pretrain_kappa.sh --stage he --num_gpus 8 \
#                                     --num_nodes 2 --master_addr X               # multi-node (manual)
#   ./scripts/train/pretrain_kappa.sh --stage he --hostfile hostfile.txt          # multi-node (hostfile)
#   ./scripts/train/pretrain_kappa.sh --stage egodex --no_deepspeed               # force plain DDP
#   ./scripts/train/pretrain_kappa.sh --stage egodex --deepspeed                  # force DeepSpeed (even single GPU)
#   ./scripts/train/pretrain_kappa.sh --stage he \
#       --train.pretrained-weights .runs/kappa/<egodex-run>                       # stage 2 from stage 1 weights
#
# Cluster scheduler (PET_*) usage:
#   The platform sets PET_NNODES / PET_NODE_RANK / MASTER_ADDR / MASTER_PORT on
#   each node before invoking this script. No flags are needed — the script
#   reads these env vars as defaults. Example submission command per node:
#       ./scripts/train/pretrain_kappa.sh --stage egodex
#   The IB / NCCL tuning block below is auto-applied when NUM_NODES > 1.
#
# Per-sample data pipeline (repack -> field -> model). Defaults: Resize 224 +
# CenterCrop 224 (~196 vision tokens with Qwen3-VL patch_size=16). Override via:
#   --data.transform.model.resize.size 256 256                                    # H W (or single int = short side)
#   --data.transform.model.center-crop.size 224                                   # square crop
#   --data.transform.model.img-aug                                                # enable ColorJitter
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON="${PROJECT_ROOT}/.venv/bin/python"
DEEPSPEED="${PROJECT_ROOT}/.venv/bin/deepspeed"
TRAIN_SCRIPT="${PROJECT_ROOT}/scripts/train.py"

STAGE=""
CONFIG_NAME=""
NUM_GPUS=""
# Pick up cluster scheduler env vars (PET_NNODES / PET_NODE_RANK) as defaults
# so this script works unchanged under torchrun-style submission systems.
NUM_NODES="${PET_NNODES:-1}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29500}"
ZERO_STAGE=2
NO_DEEPSPEED=""
DS_CONFIG=""
HOSTFILE=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage)          STAGE="$2"; shift 2 ;;
        --config)         CONFIG_NAME="$2"; shift 2 ;;
        --num_gpus)       NUM_GPUS="$2"; shift 2 ;;
        --num_nodes)      NUM_NODES="$2"; shift 2 ;;
        --node_rank)      NODE_RANK="$2"; shift 2 ;;
        --master_addr)    MASTER_ADDR="$2"; shift 2 ;;
        --master_port)    MASTER_PORT="$2"; shift 2 ;;
        --zero_stage)     ZERO_STAGE="$2"; shift 2 ;;
        --ds_config)      DS_CONFIG="$2"; shift 2 ;;
        --hostfile)       HOSTFILE="$2"; shift 2 ;;
        --no_deepspeed)   NO_DEEPSPEED=true; shift ;;
        --deepspeed)      NO_DEEPSPEED=false; shift ;;
        *)                EXTRA_ARGS+=("$1"); shift ;;
    esac
done

if [ -n "$STAGE" ] && [ -n "$CONFIG_NAME" ]; then
    echo "Error: --stage and --config are mutually exclusive" >&2
    exit 1
fi

if [ -z "$CONFIG_NAME" ]; then
    case "$STAGE" in
        egodex) CONFIG_NAME="pretrain_egodex_kappa_config" ;;
        he)     CONFIG_NAME="pretrain_he_kappa_config" ;;
        "")     echo "Error: must specify --stage {egodex|he} or --config <name>" >&2; exit 1 ;;
        *)      echo "Error: unknown --stage '$STAGE' (expected 'egodex' or 'he')" >&2; exit 1 ;;
    esac
fi

NODE_RANK="${NODE_RANK:-${PET_NODE_RANK:-0}}"

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

echo "==> Config: ${CONFIG_NAME}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_DEBUG="${NCCL_DEBUG:-ERROR}"

# Multi-node IB / NCCL tuning. These mirror the cluster's recommended
# torchrun example (RDMA over bond0, mlx5 HCA exclusions). Each var uses
# `:-` so a caller can override any single setting via the environment.
# Skipped on single-node since the IB path is unused inside one box.
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
