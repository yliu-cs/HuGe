#!/usr/bin/env bash
set -euo pipefail

# Sequential stage3 experiment runner for architecture comparisons.
#
# By default this runs stage3 once for each architecture:
#   1. qwenpi_v3_stage3
#   2. qwengr00t_stage3
#   3. qwenoft_stage3
#
# Each run starts from scratch with --init-ckpt none.
#
# Each stage calls train_kappa.sh and writes to:
#
#   ${KAPPA_RUN_ROOT:-.runs}/<experiment_name>/<stage>/<timestamp>/
#
# Useful environment variables:
#   KAPPA_DRY_RUN=1                 print commands without launching training
#   KAPPA_RUN_ROOT=.runs/xxx        change the root for all experiments
#   KAPPA_LOCK_FILE=...             override the default run flag file

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="$SCRIPT_DIR/train_kappa.sh"
RUN_ROOT="${KAPPA_RUN_ROOT:-.runs}"
DRY_RUN="${KAPPA_DRY_RUN:-0}"
RUN_COUNT=0
LOCK_FILE="${KAPPA_LOCK_FILE:-$RUN_ROOT/run_kappa.flag}"
LOCK_HOST="$(hostname 2>/dev/null || printf 'unknown')"
LOCK_TOKEN="$LOCK_HOST:$$:$(date +%s)"

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

acquire_run_lock() {
  mkdir -p "$(dirname -- "$LOCK_FILE")"

  if (
    set -o noclobber
    {
      printf 'token=%s\n' "$LOCK_TOKEN"
      printf 'host=%s\n' "$LOCK_HOST"
      printf 'pid=%s\n' "$$"
      printf 'started_at=%s\n' "$(date -Is)"
    } > "$LOCK_FILE"
  ) 2>/dev/null; then
    trap release_run_lock EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    return 0
  fi

  echo "Flag file already exists, another run_kappa.sh is running: $LOCK_FILE" >&2
  if [[ -r "$LOCK_FILE" ]]; then
    echo "Existing flag:" >&2
    sed 's/^/  /' "$LOCK_FILE" >&2 || true
  fi
  exit 0
}

release_run_lock() {
  if [[ -f "$LOCK_FILE" ]] && grep -qx "token=$LOCK_TOKEN" "$LOCK_FILE" 2>/dev/null; then
    rm -f "$LOCK_FILE"
  fi
}

run_exp() {
  local name="$1"
  local stage="$2"
  shift 2

  local out_dir="$RUN_ROOT/$name"
  local cmd=("$TRAIN_SCRIPT" "$stage" --output-dir "$out_dir")

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

run_stage3_no_init() {
  local architecture="$1"
  local name="${architecture}_stage3"

  run_exp "$name" stage3 --model-architecture "$architecture" --init-ckpt none
}

run_all() {
  local architectures=(qwenpi_v3 qwengr00t qwenoft)
  local architecture

  for architecture in "${architectures[@]}"; do
    run_stage3_no_init "$architecture"
  done
}

acquire_run_lock
run_all

if [[ "$RUN_COUNT" -eq 0 ]]; then
  cat <<EOF
No experiments are enabled.

Edit scripts/train/run_kappa.sh and add run_exp blocks inside run_all().
Use KAPPA_DRY_RUN=1 scripts/train/run_kappa.sh to preview commands.
EOF
fi
