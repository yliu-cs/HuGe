#!/usr/bin/env bash
# Evaluate the current 9-checkpoint HumanoidArena matrix in stable shards.
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: bash scripts/eval/run_ha_eval.sh <index> 2
  index must be exactly 0 or 1; the shard count is frozen to 2.

Default run-specific roots (environment variables may override):
  HUGE_EVAL_QWENPI_V3_RUN_ROOT
  HUGE_EVAL_QWENGR00T_RUN_ROOT
  HUGE_EVAL_QWENOFT_RUN_ROOT

Matrix/output overrides:
  HUGE_EVAL_EXPECTED_STEPS (default 6000,12000,18000)
  HUGE_EVAL_OUTPUT_ROOT

Optional topology overrides (defaults: 8 GPUs, 2 workers/GPU):
  HUGE_EVAL_GPU_COUNT or HUGE_EVAL_GPU_LOGICAL_IDS (mutually exclusive)
  HUGE_EVAL_WORKERS_PER_GPU=1|2
  HUGE_EVAL_MIN_FREE_GPU_MEMORY_GIB (default 47 for WPG2)
  HUGE_EVAL_MAX_CONCURRENT_SHADOW_STARTUPS (default 2)
  HUGE_EVAL_SHADOW_STARTUP_TIMEOUT_SEC (default 360)
  HUGE_ALLOW_KIT_MAP_BOOTSTRAP=0|1 (default 1)
EOF
}

if [[ "$#" -ne 2 ]]; then
  usage
  exit 2
fi
if [[ ! "$1" =~ ^(0|1)$ ]] || [[ "$2" != "2" ]]; then
  usage
  exit 2
fi

SHARD_INDEX=$((10#$1))
SHARD_COUNT=$((10#$2))

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUN_EVAL="${SCRIPT_DIR}/humanoidarena/run_eval.sh"
SELECTOR="${SCRIPT_DIR}/humanoidarena/shard_checkpoints.py"
PYTHON="${REPO_ROOT}/.venv/bin/python"
EVAL_ROOT="${HUGE_EVAL_OUTPUT_ROOT:-${REPO_ROOT}/.runs/eval/humanoidarena/full_optimized_g1_18k_20260718}"
EXPECTED_STEPS="${HUGE_EVAL_EXPECTED_STEPS:-6000,12000,18000}"
export HUGE_EVAL_QWENPI_V3_RUN_ROOT="${HUGE_EVAL_QWENPI_V3_RUN_ROOT:-${REPO_ROOT}/.runs/stage3_humanoidarena_full_optimized/HOI_pp_box/qwenpi_v3/full/stage3/20260718_170115}"
export HUGE_EVAL_QWENGR00T_RUN_ROOT="${HUGE_EVAL_QWENGR00T_RUN_ROOT:-${REPO_ROOT}/.runs/stage3_humanoidarena_full_optimized/HOI_pp_box/qwengr00t/full/stage3/20260718_170147}"
export HUGE_EVAL_QWENOFT_RUN_ROOT="${HUGE_EVAL_QWENOFT_RUN_ROOT:-${REPO_ROOT}/.runs/stage3_humanoidarena_full_optimized/HOI_pp_box/qwenoft/full/stage3/20260718_165140}"
if [[ -n "${HUGE_ALLOW_KIT_MAP_BOOTSTRAP:-}" && "${HUGE_ALLOW_KIT_MAP_BOOTSTRAP}" != "0" && "${HUGE_ALLOW_KIT_MAP_BOOTSTRAP}" != "1" ]]; then
  echo "[run_ha_eval] HUGE_ALLOW_KIT_MAP_BOOTSTRAP must be 0 or 1" >&2
  exit 2
fi
export HUGE_ALLOW_KIT_MAP_BOOTSTRAP="${HUGE_ALLOW_KIT_MAP_BOOTSTRAP:-1}"

if [[ ! -f "${SELECTOR}" ]] || [[ ! -x "${PYTHON}" ]]; then
  echo "[run_ha_eval] missing selector or Huge Python runtime" >&2
  exit 2
fi

required_root_names=(
  HUGE_EVAL_QWENPI_V3_RUN_ROOT
  HUGE_EVAL_QWENGR00T_RUN_ROOT
  HUGE_EVAL_QWENOFT_RUN_ROOT
)
for name in "${required_root_names[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "[run_ha_eval] ${name} must name one exact training run" >&2
    exit 2
  fi
done

if [[ -n "${HUGE_EVAL_GPU_COUNT:-}" && -n "${HUGE_EVAL_GPU_LOGICAL_IDS:-}" ]]; then
  echo "[run_ha_eval] HUGE_EVAL_GPU_COUNT and HUGE_EVAL_GPU_LOGICAL_IDS are mutually exclusive" >&2
  exit 2
fi
if [[ -z "${HUGE_EVAL_GPU_COUNT:-}" && -z "${HUGE_EVAL_GPU_LOGICAL_IDS:-}" ]]; then
  export HUGE_EVAL_GPU_COUNT=8
fi

if [[ -n "${HUGE_EVAL_GPU_COUNT:-}" ]]; then
  if [[ ! "${HUGE_EVAL_GPU_COUNT}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[run_ha_eval] HUGE_EVAL_GPU_COUNT must be a positive integer" >&2
    exit 2
  fi
  selected_gpu_count=$((10#${HUGE_EVAL_GPU_COUNT}))
  gpu_selection="count:${HUGE_EVAL_GPU_COUNT}"
else
  if [[ ! "${HUGE_EVAL_GPU_LOGICAL_IDS}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "[run_ha_eval] HUGE_EVAL_GPU_LOGICAL_IDS must be unique comma-separated integers" >&2
    exit 2
  fi
  IFS=',' read -r -a selected_gpu_ids <<< "${HUGE_EVAL_GPU_LOGICAL_IDS}"
  declare -A seen_gpu_ids=()
  for gpu_id in "${selected_gpu_ids[@]}"; do
    normalized_id=$((10#${gpu_id}))
    if [[ -n "${seen_gpu_ids[${normalized_id}]:-}" ]]; then
      echo "[run_ha_eval] HUGE_EVAL_GPU_LOGICAL_IDS contains duplicate ${normalized_id}" >&2
      exit 2
    fi
    seen_gpu_ids[${normalized_id}]=1
  done
  selected_gpu_count=${#selected_gpu_ids[@]}
  gpu_selection="logical_ids:${HUGE_EVAL_GPU_LOGICAL_IDS}"
fi

export HUGE_EVAL_WORKERS_PER_GPU="${HUGE_EVAL_WORKERS_PER_GPU:-2}"
if [[ "${HUGE_EVAL_WORKERS_PER_GPU}" != "1" && "${HUGE_EVAL_WORKERS_PER_GPU}" != "2" ]]; then
  echo "[run_ha_eval] HUGE_EVAL_WORKERS_PER_GPU must be 1 or 2" >&2
  exit 2
fi

if [[ "${HUGE_EVAL_WORKERS_PER_GPU}" == "2" ]]; then
  export HUGE_EVAL_MIN_FREE_GPU_MEMORY_GIB="${HUGE_EVAL_MIN_FREE_GPU_MEMORY_GIB:-47}"
  if ! "${PYTHON}" -c \
    'import math, sys; value=float(sys.argv[1]); raise SystemExit(0 if math.isfinite(value) and value >= 47.0 else 1)' \
    "${HUGE_EVAL_MIN_FREE_GPU_MEMORY_GIB}"; then
    echo "[run_ha_eval] WPG2 48GB profile requires HUGE_EVAL_MIN_FREE_GPU_MEMORY_GIB>=47" >&2
    exit 2
  fi
else
  export HUGE_EVAL_MIN_FREE_GPU_MEMORY_GIB="${HUGE_EVAL_MIN_FREE_GPU_MEMORY_GIB:-0}"
  if ! "${PYTHON}" -c \
    'import math, sys; value=float(sys.argv[1]); raise SystemExit(0 if math.isfinite(value) and value >= 0.0 else 1)' \
    "${HUGE_EVAL_MIN_FREE_GPU_MEMORY_GIB}"; then
    echo "[run_ha_eval] HUGE_EVAL_MIN_FREE_GPU_MEMORY_GIB must be finite and non-negative" >&2
    exit 2
  fi
fi

export HUGE_EVAL_OUTPUT_ROOT="${EVAL_ROOT}"
export HUGE_EVAL_FORCE_NEW_RUN=0
export HUGE_EVAL_RESUME=1
export HUGE_EVAL_DRY_RUN=0
export SEEDS_OVERRIDE=0,1,2
export REPEATS_PER_SEED=20
export MAX_STEPS=1450
export TEST_MODE=base_test
export VIDEO_FPS=30
export POST_TERMINATION_RECORD_STEPS=10
export HUGE_EVAL_RECORD_VIDEO_EVERY_N=1
export HUGE_EVAL_EPISODE_MAX_RETRIES=1
export HUGE_EVAL_MODEL_IO_TRACE_POLICY=require_complete_v1
export HUGE_EVAL_MAX_CONCURRENT_SHADOW_STARTUPS="${HUGE_EVAL_MAX_CONCURRENT_SHADOW_STARTUPS:-2}"
export HUGE_EVAL_SHADOW_STARTUP_TIMEOUT_SEC="${HUGE_EVAL_SHADOW_STARTUP_TIMEOUT_SEC:-360}"

total_workers=$((selected_gpu_count * HUGE_EVAL_WORKERS_PER_GPU))
echo "[run_ha_eval] topology gpu_selection=${gpu_selection} workers_per_gpu=${HUGE_EVAL_WORKERS_PER_GPU} total_workers=${total_workers} min_free_gib=${HUGE_EVAL_MIN_FREE_GPU_MEMORY_GIB} output_root=${EVAL_ROOT} expected_steps=${EXPECTED_STEPS} base_port=${HUGE_EVAL_BASE_PORT:-18080} kit_map_bootstrap=${HUGE_ALLOW_KIT_MAP_BOOTSTRAP} model_noise_seed_algorithm=huge_ha_model_noise_v1" >&2

CHECKPOINT_ROOTS=(
  "${HUGE_EVAL_QWENPI_V3_RUN_ROOT}"
  "${HUGE_EVAL_QWENGR00T_RUN_ROOT}"
  "${HUGE_EVAL_QWENOFT_RUN_ROOT}"
)

cd "${REPO_ROOT}"

selection_file="$(mktemp /tmp/huge-ha-eval-shard.XXXXXX)"
cleanup() {
  rm -f -- "${selection_file}"
}
trap cleanup EXIT

if ! PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${REPO_ROOT}/src" \
  "${PYTHON}" "${SELECTOR}" \
  --repo-root "${REPO_ROOT}" \
  --eval-root "${EVAL_ROOT}" \
  --expected-steps "${EXPECTED_STEPS}" \
  --shard-index "${SHARD_INDEX}" \
  --shard-count "${SHARD_COUNT}" \
  --output-format actions \
  "${CHECKPOINT_ROOTS[@]}" >"${selection_file}"; then
  exit 2
fi
mapfile -t ASSIGNED_ACTIONS <"${selection_file}"

overall_rc=0
for position in "${!ASSIGNED_ACTIONS[@]}"; do
  IFS=$'\t' read -r action checkpoint retry_hash retry_contract <<< "${ASSIGNED_ACTIONS[${position}]}"
  if [[ "${action}" != "run" && "${action}" != "retry_required" ]]; then
    echo "[run_ha_eval] selector returned an unsupported action: ${action}" >&2
    exit 2
  fi
  if [[ -z "${checkpoint}" ]]; then
    echo "[run_ha_eval] selector returned an empty checkpoint" >&2
    exit 2
  fi

  echo "[run_ha_eval] [$((position + 1))/${#ASSIGNED_ACTIONS[@]}] action=${action} checkpoint=${checkpoint}"
  if [[ "${action}" == "retry_required" ]]; then
    if [[ ! "${retry_hash}" =~ ^[0-9a-f]{64}$ || -z "${retry_contract}" ]]; then
      echo "[run_ha_eval] retry_required lacks a valid contract path/hash" >&2
      exit 2
    fi
    echo "[run_ha_eval] controlled retry previous_contract=${retry_contract} previous_hash=${retry_hash}"
    if HUGE_EVAL_FORCE_NEW_RUN=1 \
      HUGE_EVAL_RETRY_REQUIRED_CONTRACT_HASH="${retry_hash}" \
      bash "${RUN_EVAL}" pp_box "${checkpoint}" sonic; then
      continue
    else
      exit_code=$?
    fi
  elif HUGE_EVAL_FORCE_NEW_RUN=0 \
    HUGE_EVAL_RETRY_REQUIRED_CONTRACT_HASH= \
    bash "${RUN_EVAL}" pp_box "${checkpoint}" sonic; then
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
