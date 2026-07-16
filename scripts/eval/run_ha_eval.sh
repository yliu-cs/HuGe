#!/usr/bin/env bash
# Evaluate one stable shard of qwenpi_v3, qwengr00t, and qwenoft checkpoints on eight GPUs.
set -euo pipefail

usage() {
  echo "Usage: bash scripts/eval/run_ha_eval.sh <index> <machine_count>" >&2
  echo "  index is zero-based and must satisfy 0 <= index < machine_count." >&2
}

if [[ "$#" -ne 2 ]]; then
  usage
  exit 2
fi
if [[ ! "$1" =~ ^[0-9]+$ ]] || [[ ! "$2" =~ ^[0-9]+$ ]]; then
  usage
  exit 2
fi

SHARD_INDEX=$((10#$1))
SHARD_COUNT=$((10#$2))
if ((SHARD_COUNT < 1 || SHARD_INDEX >= SHARD_COUNT)); then
  usage
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUN_EVAL="${SCRIPT_DIR}/humanoidarena/run_eval.sh"
SELECTOR="${SCRIPT_DIR}/humanoidarena/shard_checkpoints.py"
PYTHON="${REPO_ROOT}/.venv/bin/python"
EVAL_ROOT="${HUGE_EVAL_OUTPUT_ROOT:-${REPO_ROOT}/.runs/eval/humanoidarena}"

export HUGE_EVAL_GPU_COUNT=8
unset HUGE_EVAL_GPU_LOGICAL_IDS
export HUGE_EVAL_WORKERS_PER_GPU=1
export HUGE_EVAL_FORCE_NEW_RUN=0
export HUGE_EVAL_RESUME=1
export HUGE_ALLOW_KIT_MAP_BOOTSTRAP=1
export HUGE_EVAL_DRY_RUN=0
export SEEDS_OVERRIDE=0,1,2
export REPEATS_PER_SEED=20
export MAX_STEPS=1450
export TEST_MODE=base_test

CHECKPOINT_ROOTS=(
  "${REPO_ROOT}/.runs/stage3_humanoidarena/HOI_pp_box/qwenpi_v3"
  "${REPO_ROOT}/.runs/stage3_humanoidarena/HOI_pp_box/qwengr00t"
  "${REPO_ROOT}/.runs/stage3_humanoidarena/HOI_pp_box/qwenoft"
)

cd "${REPO_ROOT}"

if [[ ! -f "${SELECTOR}" ]] || [[ ! -x "${PYTHON}" ]]; then
  echo "[run_ha_eval] missing selector or Huge Python runtime" >&2
  exit 2
fi

selection_file="$(mktemp /tmp/huge-ha-eval-shard.XXXXXX)"
cleanup() {
  rm -f -- "${selection_file}"
}
trap cleanup EXIT

if ! PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${REPO_ROOT}/src" \
  "${PYTHON}" "${SELECTOR}" \
  --repo-root "${REPO_ROOT}" \
  --eval-root "${EVAL_ROOT}" \
  --shard-index "${SHARD_INDEX}" \
  --shard-count "${SHARD_COUNT}" \
  "${CHECKPOINT_ROOTS[@]}" >"${selection_file}"; then
  exit 2
fi
mapfile -t ASSIGNED_CHECKPOINTS <"${selection_file}"

overall_rc=0
for position in "${!ASSIGNED_CHECKPOINTS[@]}"; do
  checkpoint="${ASSIGNED_CHECKPOINTS[${position}]}"
  echo "[run_ha_eval] [$((position + 1))/${#ASSIGNED_CHECKPOINTS[@]}] start checkpoint=${checkpoint}"
  if bash "${RUN_EVAL}" pp_box "${checkpoint}" sonic; then
    continue
  else
    exit_code=$?
  fi

  case "${exit_code}" in
    130|143)
      exit "${exit_code}"
      ;;
    *)
      overall_rc=1
      ;;
  esac
done

exit "${overall_rc}"
