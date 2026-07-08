#!/usr/bin/env bash
# Machine-specific environment fixes for HumanoidArena sonic-backend eval,
# discovered while debugging the pp_box smoke test on this host. Source this
# from run_eval.sh (or manually before calling the batch_test_*.sh scripts
# directly).
#
#   1. PYTHONPATH: sim_eval_vla.py imports task_runtime_profiles before its
#      own sys.path fix-up runs; nothing upstream puts isaaclab_twist2_g1/ on
#      PYTHONPATH for the sim subprocess.
#   2. OMNI_KIT_ACCEPT_EULA: Isaac Sim's EULA prompt calls input(), which
#      raises EOFError with no TTY (we launch it from a subprocess pipeline).
#   3. libGLU.so.1 / libXt.so.6: required by the RTX renderer; if missing,
#      Isaac Sim segfaults ~10s after startup. `apt-get install libglu1-mesa
#      libxt6` on this host.
#   4. onnxruntime-gpu: requirements.txt pins the CPU-only `onnxruntime`
#      package; SonicActionProvider requires a GPU provider. `pip uninstall
#      onnxruntime && pip install onnxruntime-gpu==1.22.0` in unitree_sim_env.
#   5. LD_LIBRARY_PATH: TensorRT and cuDNN libs are on disk but not on the
#      loader path, so onnxruntime silently falls back to CPU.
# preflight_check (below) verifies #3/#4 instead of silently fixing them --
# those are one-time package installs, not per-run env state.
set -uo pipefail

EVAL_SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HUGE_ROOT="$(cd "${EVAL_SCRIPTS_DIR}/../../.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-${HUGE_ROOT}/HumanoidArena}"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-${PROJECT_ROOT}/isaaclab_twist2_g1}"
export ISAACLAB_ROOT

source "${ISAACLAB_ROOT}/script/common/runtime_paths.sh"  # -> CONDA_BASE, ISAACLAB_PYTHON, ISAACLAB_CONDA_ENV_NAME, ...

# --- fix #1 ---
export PYTHONPATH="${ISAACLAB_ROOT}:${PYTHONPATH:-}"

# --- fix #2 ---
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-Y}"

# --- fix #5 ---
TENSORRT_LIB="${TensorRT_ROOT:-$HOME/TensorRT}/targets/x86_64-linux-gnu/lib"
CUDNN_LIB="$(compgen -G "${CONDA_BASE}/envs/${ISAACLAB_CONDA_ENV_NAME}/lib/python3.*/site-packages/nvidia/cudnn/lib" | head -1 || true)"
[[ -d "${TENSORRT_LIB}" ]] && export LD_LIBRARY_PATH="${TENSORRT_LIB}:${LD_LIBRARY_PATH:-}"
[[ -n "${CUDNN_LIB}" && -d "${CUDNN_LIB}" ]] && export LD_LIBRARY_PATH="${CUDNN_LIB}:${LD_LIBRARY_PATH:-}"

preflight_check() {
  local ok=1

  if ! ldconfig -p 2>/dev/null | grep -q libGLU.so.1; then
    echo "[preflight] missing libGLU.so.1 -- Isaac Sim's RTX renderer will segfault. Fix: apt-get install -y libglu1-mesa libxt6" >&2
    ok=0
  fi
  if ! ldconfig -p 2>/dev/null | grep -q libXt.so.6; then
    echo "[preflight] missing libXt.so.6 -- Isaac Sim's RTX renderer will segfault. Fix: apt-get install -y libglu1-mesa libxt6" >&2
    ok=0
  fi

  if [[ ! -d "${TENSORRT_LIB}" ]]; then
    echo "[preflight] TensorRT libs not found at ${TENSORRT_LIB} -- onnxruntime will fall back to CPU. Set TensorRT_ROOT if installed elsewhere." >&2
    ok=0
  fi
  if [[ -z "${CUDNN_LIB}" ]]; then
    echo "[preflight] cuDNN libs not found under ${CONDA_BASE}/envs/${ISAACLAB_CONDA_ENV_NAME} -- onnxruntime will fall back to CPU. Fix: pip install nvidia-cudnn-cu12 in that env." >&2
    ok=0
  fi

  local providers
  providers="$("${ISAACLAB_PYTHON}" -c 'import onnxruntime as ort; print(ort.get_available_providers())' 2>/dev/null || true)"
  if [[ "${providers}" != *CUDAExecutionProvider* ]]; then
    echo "[preflight] onnxruntime has no CUDAExecutionProvider (available_providers=${providers:-<failed to query>}). Fix: pip uninstall onnxruntime && pip install onnxruntime-gpu==1.22.0 in ${ISAACLAB_CONDA_ENV_NAME}." >&2
    ok=0
  fi

  if [[ "${ok}" != 1 ]]; then
    echo "[preflight] one or more checks failed -- eval will likely crash or silently run on CPU. Fix the above, or set SKIP_PREFLIGHT=1 to proceed anyway." >&2
    [[ "${SKIP_PREFLIGHT:-0}" == "1" ]] || exit 1
  fi
}
