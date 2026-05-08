#!/usr/bin/env bash
# Kappa Fine-Tune Launcher
# Continues training the post-train Kappa checkpoint on a single HE episode
# (default ep 7831). Wraps pretrain_kappa.sh with the fine-tune config.
# Pretrained weights come from one of:
#   - the calling pipeline (run_kappa_pipeline.sh injects the previous
#     stage's run dir as --train.pretrained-weights)
#   - the user's own --train.pretrained-weights flag
#   - the config's pretrained_weights default
#
# Usage:
#   ./scripts/train/finetune_kappa.sh                                           # config default seeds the run
#   ./scripts/train/finetune_kappa.sh --num_gpus 1
#   ./scripts/train/finetune_kappa.sh \
#     --train.pretrained-weights .runs/kappa/<your-posttrain-run>               # explicit ckpt
#   ./scripts/train/finetune_kappa.sh \
#     --data.episode-indices 1234                                               # different HE episode
#
# Internally calls pretrain_kappa.sh --config finetune_he_ep_kappa_config so
# all of pretrain_kappa.sh's torchrun / DeepSpeed / multi-node tuning flags
# work unchanged.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec "$SCRIPT_DIR/pretrain_kappa.sh" \
    --config finetune_he_ep_kappa_config \
    "$@"
