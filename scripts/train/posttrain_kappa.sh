#!/usr/bin/env bash
# Kappa Post-Train Launcher
# Wraps pretrain_kappa.sh with the post-train config and a default
# pretrained-weights pointer to the stage 1-2 ckpt. Override the ckpt path
# via the --pretrained_weights flag below or by editing the config.
#
# Usage:
#   ./scripts/train/posttrain_kappa.sh                                       # 8 GPUs DeepSpeed by default
#   ./scripts/train/posttrain_kappa.sh --num_gpus 4
#   ./scripts/train/posttrain_kappa.sh \
#     --train.pretrained-weights .runs/kappa/<your-stage1-2-run>             # override ckpt
#
# Internally calls pretrain_kappa.sh --config posttrain_he_wbt_kappa_config
# so all of pretrain_kappa.sh's torchrun / DeepSpeed / multi-node tuning
# flags work unchanged.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# By default, point at the stage 1-2 ckpt the user just trained.
DEFAULT_PRETRAINED=".runs/kappa/exp.kappa.b1024.gpus8.2605070240/checkpoints/ckpt_6931"

# Allow callers to inject `--train.pretrained-weights` themselves; if they
# don't, prepend the default.
HAS_PRETRAINED=0
for arg in "$@"; do
    case "$arg" in
        --train.pretrained-weights*|--train.pretrained_weights*) HAS_PRETRAINED=1 ;;
    esac
done

EXTRA=()
if [ $HAS_PRETRAINED -eq 0 ]; then
    EXTRA+=(--train.pretrained-weights "$DEFAULT_PRETRAINED")
fi

exec "$SCRIPT_DIR/pretrain_kappa.sh" \
    --config posttrain_he_wbt_kappa_config \
    "${EXTRA[@]}" \
    "$@"
