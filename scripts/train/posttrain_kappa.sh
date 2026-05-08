#!/usr/bin/env bash
# Kappa Post-Train Launcher
# Wraps pretrain_kappa.sh with the post-train config. Pretrained weights
# come from one of:
#   - the calling pipeline (run_kappa_pipeline.sh injects the previous
#     stage's run dir as --train.pretrained-weights)
#   - the user's own --train.pretrained-weights flag
#   - the config's pretrained_weights default
#
# Usage:
#   ./scripts/train/posttrain_kappa.sh                                       # config default seeds the run
#   ./scripts/train/posttrain_kappa.sh --num_gpus 4
#   ./scripts/train/posttrain_kappa.sh \
#     --train.pretrained-weights .runs/kappa/<your-stage1-2-run>             # explicit ckpt
#
# Internally calls pretrain_kappa.sh --config posttrain_he_wbt_kappa_config
# so all of pretrain_kappa.sh's torchrun / DeepSpeed / multi-node tuning
# flags work unchanged.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec "$SCRIPT_DIR/pretrain_kappa.sh" \
    --config posttrain_he_wbt_kappa_config \
    "$@"
