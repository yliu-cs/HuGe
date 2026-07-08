#!/usr/bin/env bash
# HumanoidArena full VLA evaluation launcher.
#
# Usage: scripts/eval/humanoidarena/run_eval.sh <task> <checkpoint_dir> [backend]
#   task            boxing | doubledesk | football | open_door | pp_box | sit_sofa | vision_navi
#   checkpoint_dir  a pretrained_model dir, or a parent dir containing one
#                   (see model_batch_utils.sh's discover_model_paths)
#   backend         all (default) | sonic | twist2
#
# Runs the task's FULL eval config (all seeds x repeats_per_seed from its
# tasks/common_test_config/*/<task>_*_test.yaml) -- this is not a smoke test.
# Override via env vars as usual, e.g.:
#   TEST_MODE=base_test SEEDS_OVERRIDE=0 REPEATS_PER_SEED=1 NUM_WORKERS=1 \
#     scripts/eval/humanoidarena/run_eval.sh pp_box \
#     HumanoidArena/checkpoints/small/HOI_pp_box/diffusion_sonic_ppbox_0529
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

TASK="${1:?usage: $0 <task> <checkpoint_dir> [backend]}"
CHECKPOINT_DIR="${2:?usage: $0 <task> <checkpoint_dir> [backend]}"
export EVAL_BACKEND="${3:-${EVAL_BACKEND:-all}}"

TASK_SCRIPT="${ISAACLAB_ROOT}/batch_test_scripts/task/batch_test_${TASK}.sh"
if [[ ! -f "${TASK_SCRIPT}" ]]; then
  echo "[run_eval] unknown task '${TASK}' (no such script ${TASK_SCRIPT})" >&2
  echo "[run_eval] available tasks:" >&2
  ls "${ISAACLAB_ROOT}/batch_test_scripts/task/" | sed -n 's/^batch_test_\(.*\)\.sh$/  \1/p' >&2
  exit 2
fi
[[ -d "${CHECKPOINT_DIR}" ]] || { echo "[run_eval] checkpoint dir not found: ${CHECKPOINT_DIR}" >&2; exit 2; }
export MODEL_ROOT="$(cd "${CHECKPOINT_DIR}" && pwd)"

preflight_check

echo "[run_eval] task=${TASK} backend=${EVAL_BACKEND} model_root=${MODEL_ROOT}"
exec bash "${TASK_SCRIPT}"
