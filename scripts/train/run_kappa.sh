#!/usr/bin/env bash
set -euo pipefail

# Sequential experiment runner.
#
# Edit run_all() at the bottom of this file and add one run_exp block per
# experiment. Each experiment calls train_kappa.sh and writes to:
#
#   ${KAPPA_RUN_ROOT:-.runs}/<experiment_name>/<stage>/<timestamp>/
#
# Useful environment variables:
#   KAPPA_DRY_RUN=1                 print commands without launching training
#   KAPPA_RUN_ROOT=.runs/xxx        change the root for all experiments
#   KAPPA_DEFAULT_INIT_CKPT=...     shared init checkpoint for stage3 experiments

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="$SCRIPT_DIR/train_kappa.sh"
RUN_ROOT="${KAPPA_RUN_ROOT:-.runs}"
DRY_RUN="${KAPPA_DRY_RUN:-0}"
DEFAULT_STAGE="${KAPPA_DEFAULT_STAGE:-stage3}"
DEFAULT_INIT_CKPT="${KAPPA_DEFAULT_INIT_CKPT:-}"
RUN_COUNT=0

COMMON_ARGS=(
  # Add args shared by every experiment here, for example:
  # --batch-size 32
  # --num-workers 8
)

truthy() {
  case "${1,,}" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

has_init_ckpt_arg() {
  local arg
  for arg in "$@"; do
    case "$arg" in
      --init-ckpt|--init_ckpt|--init-ckpt=*|--init_ckpt=*) return 0 ;;
    esac
  done
  return 1
}

run_exp() {
  local name="$1"
  shift

  local stage="$DEFAULT_STAGE"
  if [[ $# -gt 0 && "$1" != --* ]]; then
    stage="$1"
    shift
  fi

  local out_dir="$RUN_ROOT/$name"
  local cmd=("$TRAIN_SCRIPT" "$stage" --output-dir "$out_dir")

  if [[ -n "$DEFAULT_INIT_CKPT" && "$stage" == "stage3" ]] && ! has_init_ckpt_arg "$@"; then
    cmd+=(--init-ckpt "$DEFAULT_INIT_CKPT")
  fi

  cmd+=("${COMMON_ARGS[@]}" "$@")

  RUN_COUNT=$((RUN_COUNT + 1))
  echo
  echo "==> [$RUN_COUNT] $name"
  printf '==> command:'
  printf ' %q' "${cmd[@]}"
  echo

  if truthy "$DRY_RUN"; then
    return 0
  fi

  "${cmd[@]}"
}

run_all() {
  # Add/edit experiments below. They run sequentially in the order listed.
  #
  # Example:
  #
  # run_exp prop_baseline stage3 \
  #   --prop-roots resource/dataset/proprietary/real/cola_merged_cleaned_0530 \
  #   --batch-size 32
  #
  # run_exp prop_video_1to1 stage3 \
  #   --prop-roots resource/dataset/proprietary/real/cola_merged_cleaned_0530 \
  #   --batch-size 32
  :

  run_exp full_param_train stage1
  run_exp full_param_train stage2
  run_exp full_param_train stage3

  run_exp lora_freeze_train stage1 \
    --freeze-qwen \
    --qwen-lora \
    --train-qwen-lora
  run_exp lora_freeze_train stage2 \
    --freeze-qwen \
    --qwen-lora \
    --train-qwen-lora
  run_exp lora_freeze_train stage3 \
    --freeze-qwen \
    --qwen-lora \
    --no-train-qwen-lora
}

run_all

if [[ "$RUN_COUNT" -eq 0 ]]; then
  cat <<EOF
No experiments are enabled.

Edit scripts/train/run_kappa.sh and add run_exp blocks inside run_all().
Use KAPPA_DRY_RUN=1 scripts/train/run_kappa.sh to preview commands.
EOF
fi
