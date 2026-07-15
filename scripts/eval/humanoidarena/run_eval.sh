#!/usr/bin/env bash
# Huge-owned HumanoidArena SONIC evaluation launcher.
# HumanoidArena is an immutable runtime dependency; all outputs stay in Huge.
set -euo pipefail

usage() {
  echo "usage: $0 <task> <checkpoint_dir_or_root> [sonic]" >&2
  echo "tasks: boxing doubledesk football open_door pp_box sit_sofa vision_navi" >&2
}

if (( $# < 2 || $# > 3 )); then
  usage
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

set +e
source "${SCRIPT_DIR}/env.sh"
ENV_RC=$?
set -e
if (( ENV_RC != 0 )); then
  echo "[run_eval] failed to resolve the HumanoidArena/Isaac Lab environment" >&2
  exit 2
fi

TASK="$1"
CHECKPOINT_DIR="$2"
BACKEND="${3:-sonic}"

case "${TASK}" in
  boxing|doubledesk|football|open_door|pp_box|sit_sofa|vision_navi) ;;
  *)
    echo "[run_eval] unknown task: ${TASK}" >&2
    usage
    exit 2
    ;;
esac

if [[ "${BACKEND}" != "sonic" ]]; then
  echo "[run_eval] backend must be sonic; all/twist2 are not supported" >&2
  exit 2
fi

if [[ "${HUGE_EVAL_DRY_RUN:-0}" != "0" && "${HUGE_EVAL_DRY_RUN:-0}" != "1" ]]; then
  echo "[run_eval] HUGE_EVAL_DRY_RUN must be 0 or 1" >&2
  exit 2
fi

if [[ ! -d "${PROJECT_ROOT}" || ! -d "${ISAACLAB_ROOT}" || ! -x "${ISAACLAB_PYTHON}" ]]; then
  echo "[run_eval] HumanoidArena or Isaac Lab runtime path is missing" >&2
  exit 2
fi

HUMANOIDARENA_ROOT="$(cd "${PROJECT_ROOT}" && pwd)"
ISAACLAB_ROOT_RESOLVED="$(cd "${ISAACLAB_ROOT}" && pwd)"
ISAAC_PYTHONPATH="${PYTHONPATH:-}"

if [[ "${HUMANOIDARENA_ROOT}" != "${REPO_ROOT}/HumanoidArena" ]]; then
  echo "[run_eval] PROJECT_ROOT must be the repository-local HumanoidArena" >&2
  exit 2
fi
if [[ "${ISAACLAB_ROOT_RESOLVED}" != "${HUMANOIDARENA_ROOT}/isaaclab_twist2_g1" ]]; then
  echo "[run_eval] ISAACLAB_ROOT must be HumanoidArena/isaaclab_twist2_g1" >&2
  exit 2
fi
if [[ ! -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  echo "[run_eval] Huge root virtual environment is missing" >&2
  exit 2
fi

cd "${REPO_ROOT}"
if [[ ! -d "${CHECKPOINT_DIR}" ]]; then
  echo "[run_eval] checkpoint directory not found: ${CHECKPOINT_DIR}" >&2
  exit 2
fi

CHECKPOINT_BASENAME="$(basename -- "${CHECKPOINT_DIR%/}")"
if [[ ! "${CHECKPOINT_BASENAME}" =~ ^step_[0-9]+$ ]]; then
  exec env \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPYCACHEPREFIX=/tmp/huge-agent-pycache \
    PYTHONPATH="${REPO_ROOT}/src" \
    "${REPO_ROOT}/.venv/bin/python" \
    "${SCRIPT_DIR}/batch_eval.py" \
    "${TASK}" "${CHECKPOINT_DIR}" "${BACKEND}"
fi

if [[ "${HUGE_EVAL_DRY_RUN:-0}" != "1" ]]; then
  if ! (preflight_check); then
    exit 2
  fi
fi

exec env \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPYCACHEPREFIX=/tmp/huge-agent-pycache \
  PYTHONPATH="${REPO_ROOT}/src" \
  "${REPO_ROOT}/.venv/bin/python" \
  "${SCRIPT_DIR}/eval_orchestrator.py" \
  --humanoidarena-root "${HUMANOIDARENA_ROOT}" \
  --isaaclab-root "${ISAACLAB_ROOT_RESOLVED}" \
  --isaaclab-python "${ISAACLAB_PYTHON}" \
  --isaac-pythonpath "${ISAAC_PYTHONPATH}" \
  "${TASK}" "${CHECKPOINT_DIR}" "${BACKEND}"
