#!/usr/bin/env bash
set -euo pipefail

# Experiment runner for the default HumanoidArena three-architecture full-only
# G1 recipe, with legacy tuning modes and sim/real pipelines available only
# through explicit environment overrides.
#
# sim mode (HUGE_STAGE3_DATA=sim) runs one qwenpi_v3 pipeline per stage1 stride:
#   stage1: EgoDex pretraining with sample_stride in {1,2,5,10,50}
#   stage2: HE + WBT posttraining, initialized from stage1
#   stage3: G1WholebodyPushOfficeChairTeleop sim finetuning, initialized from stage2
#   writes to ${HUGE_RUN_ROOT:-./.runs/stage1_stride_abla}/stride_<stride>/<stage>/<ts>/
#
# real mode (HUGE_STAGE3_DATA=real) reuses the existing sim stage1/stage2 and
# only runs a real-robot stage3 per stride, initialized from the stage2 ckpt
# found under HUGE_STAGE2_SOURCE_ROOT. The sim tree is never written to:
#   writes to ${HUGE_RUN_ROOT:-./.runs/stage3_real_cola}/stride_<stride>/stage3/<ts>/
#
# humanoid_arena mode (the default) trains all tasks with full tuning. Without an
# architecture selector it uses the fixed order qwenoft -> qwenpi_v3 ->
# qwengr00t; formal multi-host runs select exactly one architecture with
# --arch. Add --full-pipeline to run stage1 -> stage2 -> stage3 per architecture.
# It writes to:
#   ${HUGE_RUN_ROOT:-./.runs/full_all_stage}/
#     all/<arch>/full/stage3/<run_id>/
#
# Useful environment variables:
#   HUGE_DRY_RUN=1                 print commands without launching training
#   HUGE_STAGE3_DATA=sim|real|humanoid_arena   experiment mode (default humanoid_arena)
#   HUGE_RUN_ROOT=.runs/xxx        change the root for all experiments
#   HUGE_STAGE2_SOURCE_ROOT=...    (real) existing stage1/stage2 runs to init from
#   HUGE_STAGE1_SAMPLE_STRIDES=... space-separated stride values
#   HUGE_SIM_STAGE3_EPOCHS=...     sim stage3 epochs (default 50)
#   HUGE_SIM_STAGE3_CHECKPOINTING_STEPS=...   sim stage3 ckpt interval (default 500)
#   HUGE_REAL_STAGE3_EPOCHS=...    real stage3 epochs (default 50)
#   HUGE_REAL_STAGE3_CHECKPOINTING_STEPS=...  real stage3 ckpt interval (default 20000)
#   HUGE_HUMANOID_ARENA_TASK=...   exact HumanoidArena task name, or all (default all)
#   HUGE_HUMANOID_ARENA_INIT_CKPT=...  optional init checkpoint (default none/from scratch)
#   HUGE_STAGE12_DATA_ROOT=...           Stage 1/2 dataset root (default resource/datasets)
#   HUGE_STAGE2_PROPRIO_STATS=...        Stage 2 stats under that root by default
#   HUGE_STAGE12_VLM_LR=...               Stage 1/2 VLM LR (default 2e-5)
#   HUGE_STAGE12_PROJECTOR_LR=...         Stage 1/2 projector LR (default 5e-5)
#   HUGE_STAGE12_ACTION_LR=...            Stage 1/2 action-head LR (default 2e-4)
#   HUGE_HUMANOID_ARENA_ARCHITECTURES=...     space-separated archs (default "qwenoft qwenpi_v3 qwengr00t")
#   HUGE_HUMANOID_ARENA_TUNING_MODES=...      space-separated modes (default "full")
#   HUGE_HUMANOID_ARENA_RESUME_STATE=...      resume state; requires one architecture and full mode
#   HUGE_HUMANOID_ARENA_VLM_LR=...            full VLM LR (default 1e-5)
#   HUGE_HUMANOID_ARENA_PROJECTOR_LR=...      full projector LR (default 2.5e-5)
#   HUGE_HUMANOID_ARENA_ACTION_LR=...         full action-head LR (default 1e-4)
#   HUGE_HUMANOID_ARENA_WARMUP_RATIO=...      full warmup ratio (default 0.05)
#   HUGE_HUMANOID_ARENA_MIN_LR_RATE=...       full cosine floor ratio (default 0.04)
#   HUGE_HUMANOID_ARENA_CHECKPOINTING_STEPS=... full model/state interval (default 20000)
#   HUGE_HUMANOID_ARENA_MAX_STEPS=...             shared full hard cap (default 140000)
#   HUGE_HUMANOID_ARENA_QWENOFT_MAX_STEPS=...    optional qwenoft override
#   HUGE_HUMANOID_ARENA_QWENPI_V3_MAX_STEPS=...  optional qwenpi_v3 override
#   HUGE_HUMANOID_ARENA_QWENGR00T_MAX_STEPS=...  optional qwengr00t override
#   HUGE_HUMANOID_ARENA_LORA_R=...            language-attention LoRA rank (default 8)
#   HUGE_HUMANOID_ARENA_LORA_ALPHA=...        language-attention LoRA alpha (default 16)
#   HUGE_HUMANOID_ARENA_LORA_DROPOUT=...      language-attention LoRA dropout (default 0.0)
#   HUGE_HUMANOID_ARENA_STAGE3_EPOCHS=...      humanoid_arena stage3 epochs (default 50)
#   HUGE_HUMANOID_ARENA_STAGE3_CHECKPOINTING_STEPS=...  legacy alias for checkpoint interval
#   HUGE_MODEL_ARCHITECTURE=...    qwenpi_v3, qwengr00t, or qwenoft (sim/real modes only)
#   HUGE_LOCK_FILE=...             override the default run flag file
#
# Usage:
#   scripts/train/run_huge.sh --full-pipeline   run all three full pipelines sequentially
#   scripts/train/run_huge.sh --full-pipeline --arch qwenoft  run one formal 8-GPU pipeline
#   scripts/train/run_huge.sh --arch qwenoft    run one Stage 3-only architecture
#   scripts/train/run_huge.sh                   run all three architectures sequentially
#   HUGE_STAGE3_DATA=sim scripts/train/run_huge.sh   run all configured sim strides
#   scripts/train/run_huge.sh 10           run only stride_10
#   scripts/train/run_huge.sh --stride 10  run only stride_10

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="$SCRIPT_DIR/train_huge.sh"
# Experiment mode: "sim" runs the full stage1->stage2->stage3 sim pipeline;
# "real" skips stage1/stage2 and finetunes a real-robot stage3 from existing
# stage2 checkpoints (see STAGE2_SOURCE_ROOT below).
STAGE3_DATA="${HUGE_STAGE3_DATA:-humanoid_arena}"
case "$STAGE3_DATA" in
  sim)  DEFAULT_RUN_ROOT="./.runs/stage1_stride_abla" ;;
  real) DEFAULT_RUN_ROOT="./.runs/stage3_real_cola" ;;
  humanoid_arena) DEFAULT_RUN_ROOT="./.runs/full_all_stage" ;;
  *) echo "HUGE_STAGE3_DATA must be 'sim', 'real', or 'humanoid_arena', got '$STAGE3_DATA'" >&2; exit 2 ;;
esac
RUN_ROOT="${HUGE_RUN_ROOT:-$DEFAULT_RUN_ROOT}"
MODEL_ARCHITECTURE="${HUGE_MODEL_ARCHITECTURE:-qwenpi_v3}"
DRY_RUN="${HUGE_DRY_RUN:-0}"
RUN_COUNT=0
LOCK_HOST="$(hostname 2>/dev/null || printf 'unknown')"
LOCK_TOKEN="$LOCK_HOST:$$:$(date +%s)"

# --- sim stage3 (G1 teleop sim finetuning, tuned for the ~12k-frame sim set) ---
SIM_ROOT="${HUGE_STAGE3_SIM_ROOT:-resource/dataset/proprietary/sim/G1WholebodyPushOfficeChairTeleop}"
SIM_STATS="${HUGE_STAGE3_SIM_PROPRIO_STATS:-$SIM_ROOT/meta/proprio_stats_union_sonic.json}"
SIM_STAGE3_EPOCHS="${HUGE_SIM_STAGE3_EPOCHS:-50}"
SIM_STAGE3_CHECKPOINTING_STEPS="${HUGE_SIM_STAGE3_CHECKPOINTING_STEPS:-500}"

# --- real stage3 (cola real-robot finetuning, tuned for the ~433k-frame real set) ---
REAL_ROOT="${HUGE_STAGE3_REAL_ROOT:-resource/dataset/proprietary/real/cola_merged_cleaned_0608_384}"
REAL_STATS="${HUGE_STAGE3_REAL_PROPRIO_STATS:-$REAL_ROOT/meta/proprio_stats_union_sonic.json}"
REAL_STAGE3_EPOCHS="${HUGE_REAL_STAGE3_EPOCHS:-50}"
REAL_STAGE3_CHECKPOINTING_STEPS="${HUGE_REAL_STAGE3_CHECKPOINTING_STEPS:-20000}"

# Existing stage1/stage2 runs used to initialize real stage3 (read-only).
STAGE2_SOURCE_ROOT="${HUGE_STAGE2_SOURCE_ROOT:-./.runs/stage1_stride_abla}"

# --- humanoid_arena stage3 (full-only by default; legacy modes are opt-in) ---
HUMANOID_ARENA_TASK="${HUGE_HUMANOID_ARENA_TASK:-all}"
HUMANOID_ARENA_ROOT="${HUGE_STAGE3_HUMANOID_ARENA_ROOT:-resource/datasets/HumanoidArena}"
HUMANOID_ARENA_STATS="${HUGE_STAGE3_HUMANOID_ARENA_PROPRIO_STATS:-$HUMANOID_ARENA_ROOT/meta/proprio_stats_union_sonic.json}"
HUMANOID_ARENA_INIT_CKPT="${HUGE_HUMANOID_ARENA_INIT_CKPT:-none}"
HUMANOID_ARENA_RESUME_STATE="${HUGE_HUMANOID_ARENA_RESUME_STATE:-}"
HUMANOID_ARENA_STAGE3_EPOCHS="${HUGE_HUMANOID_ARENA_STAGE3_EPOCHS:-50}"
HUMANOID_ARENA_CHECKPOINTING_STEPS="${HUGE_HUMANOID_ARENA_CHECKPOINTING_STEPS:-${HUGE_HUMANOID_ARENA_STAGE3_CHECKPOINTING_STEPS:-20000}}"
HUMANOID_ARENA_RESUME_STATE_KEEP="${HUGE_HUMANOID_ARENA_RESUME_STATE_KEEP:-2}"
HUMANOID_ARENA_VLM_LR="${HUGE_HUMANOID_ARENA_VLM_LR:-1e-5}"
HUMANOID_ARENA_PROJECTOR_LR="${HUGE_HUMANOID_ARENA_PROJECTOR_LR:-2.5e-5}"
HUMANOID_ARENA_ACTION_LR="${HUGE_HUMANOID_ARENA_ACTION_LR:-1e-4}"
HUMANOID_ARENA_WARMUP_RATIO="${HUGE_HUMANOID_ARENA_WARMUP_RATIO:-0.05}"
HUMANOID_ARENA_MIN_LR_RATE="${HUGE_HUMANOID_ARENA_MIN_LR_RATE:-0.04}"
HUMANOID_ARENA_MAX_STEPS="${HUGE_HUMANOID_ARENA_MAX_STEPS:-140000}"
HUMANOID_ARENA_QWENOFT_MAX_STEPS="${HUGE_HUMANOID_ARENA_QWENOFT_MAX_STEPS:-$HUMANOID_ARENA_MAX_STEPS}"
HUMANOID_ARENA_QWENPI_V3_MAX_STEPS="${HUGE_HUMANOID_ARENA_QWENPI_V3_MAX_STEPS:-$HUMANOID_ARENA_MAX_STEPS}"
HUMANOID_ARENA_QWENGR00T_MAX_STEPS="${HUGE_HUMANOID_ARENA_QWENGR00T_MAX_STEPS:-$HUMANOID_ARENA_MAX_STEPS}"
HUMANOID_ARENA_LEGACY_LR="${HUGE_HUMANOID_ARENA_LEARNING_RATE:-3e-5}"
read -r -a HUMANOID_ARENA_ARCHITECTURES <<< "${HUGE_HUMANOID_ARENA_ARCHITECTURES:-qwenoft qwenpi_v3 qwengr00t}"
read -r -a HUMANOID_ARENA_TUNING_MODES <<< "${HUGE_HUMANOID_ARENA_TUNING_MODES:-full}"
HUMANOID_ARENA_LORA_R="${HUGE_HUMANOID_ARENA_LORA_R:-8}"
HUMANOID_ARENA_LORA_ALPHA="${HUGE_HUMANOID_ARENA_LORA_ALPHA:-16}"
HUMANOID_ARENA_LORA_DROPOUT="${HUGE_HUMANOID_ARENA_LORA_DROPOUT:-0.0}"
STAGE12_DATA_ROOT="${HUGE_STAGE12_DATA_ROOT:-resource/datasets}"
STAGE2_PROPRIO_STATS="${HUGE_STAGE2_PROPRIO_STATS:-$STAGE12_DATA_ROOT/G1_WBT-cache/meta/g1_proprio_stats_sonic.json}"
STAGE12_VLM_LR="${HUGE_STAGE12_VLM_LR:-2e-5}"
STAGE12_PROJECTOR_LR="${HUGE_STAGE12_PROJECTOR_LR:-5e-5}"
STAGE12_ACTION_LR="${HUGE_STAGE12_ACTION_LR:-2e-4}"

read -r -a STAGE1_SAMPLE_STRIDES <<< "${HUGE_STAGE1_SAMPLE_STRIDES:-1 2 5 10 50}"
SELECTED_STRIDE=""
SELECTED_ARCH=""
FULL_PIPELINE=0

usage() {
  cat <<EOF
Usage: $0 [--full-pipeline] [--arch ARCH]
       $0 [--stride STRIDE] [STRIDE]

For HumanoidArena, --arch must be one of qwenoft, qwenpi_v3, or qwengr00t.
Without --arch, run all three architectures. --full-pipeline adds Stage 1 and
Stage 2 before HumanoidArena Stage 3 for each architecture.
With HUGE_STAGE3_DATA=sim|real, optionally select one stride; configured strides:
${STAGE1_SAMPLE_STRIDES[*]}
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --full-pipeline)
      FULL_PIPELINE=1
      shift
      ;;
    --arch)
      [[ $# -ge 2 && -n "$2" ]] || { echo "--arch requires a value" >&2; exit 2; }
      [[ -z "$SELECTED_ARCH" ]] || { echo "--arch may be specified only once" >&2; exit 2; }
      SELECTED_ARCH="$2"
      shift 2
      ;;
    --arch=*)
      [[ -z "$SELECTED_ARCH" ]] || { echo "--arch may be specified only once" >&2; exit 2; }
      SELECTED_ARCH="${1#*=}"
      [[ -n "$SELECTED_ARCH" ]] || { echo "--arch requires a value" >&2; exit 2; }
      shift
      ;;
    --stride)
      [[ $# -ge 2 ]] || { echo "--stride requires a value" >&2; exit 2; }
      SELECTED_STRIDE="$2"
      shift 2
      ;;
    --stride=*)
      SELECTED_STRIDE="${1#*=}"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      if [[ -n "$SELECTED_STRIDE" ]]; then
        echo "only one stride can be selected per run_huge.sh process" >&2
        exit 2
      fi
      SELECTED_STRIDE="$1"
      shift
      ;;
  esac
done

if [[ -n "$SELECTED_STRIDE" ]]; then
  STAGE1_SAMPLE_STRIDES=("$SELECTED_STRIDE")
fi

if [[ -n "$SELECTED_ARCH" ]]; then
  if [[ "$STAGE3_DATA" != "humanoid_arena" ]]; then
    echo "--arch is only valid with HUGE_STAGE3_DATA=humanoid_arena" >&2
    exit 2
  fi
  case "$SELECTED_ARCH" in
    qwenoft|qwenpi_v3|qwengr00t) ;;
    *) echo "invalid HumanoidArena architecture: $SELECTED_ARCH" >&2; exit 2 ;;
  esac
  HUMANOID_ARENA_ARCHITECTURES=("$SELECTED_ARCH")
fi
if [[ "$STAGE3_DATA" == "humanoid_arena" && -n "$SELECTED_STRIDE" ]]; then
  echo "--stride is not valid with HUGE_STAGE3_DATA=humanoid_arena" >&2
  exit 2
fi

if [[ "$STAGE3_DATA" == "humanoid_arena" ]] &&
   [[ "${#HUMANOID_ARENA_ARCHITECTURES[@]}" -eq 1 ]] &&
   [[ -z "${HUGE_ACCELERATE_CONFIG_DIR:-}" ]]; then
  export HUGE_ACCELERATE_CONFIG_DIR="$RUN_ROOT/_accelerate/${HUMANOID_ARENA_ARCHITECTURES[0]}"
fi

LOCK_SCOPE="all"
if [[ "$STAGE3_DATA" == "humanoid_arena" ]]; then
  LOCK_SCOPE="humanoid_arena_${HUMANOID_ARENA_TASK}"
  if [[ "${#HUMANOID_ARENA_ARCHITECTURES[@]}" -eq 1 ]]; then
    LOCK_SCOPE="${LOCK_SCOPE}_${HUMANOID_ARENA_ARCHITECTURES[0]}"
  fi
elif [[ "${#STAGE1_SAMPLE_STRIDES[@]}" -eq 1 ]]; then
  LOCK_SCOPE="stride_${STAGE1_SAMPLE_STRIDES[0]}"
fi
LOCK_FILE="${HUGE_LOCK_FILE:-$RUN_ROOT/run_huge_${LOCK_SCOPE}.flag}"

truthy() {
  case "${1,,}" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

if truthy "$FULL_PIPELINE" && [[ "$STAGE3_DATA" != "humanoid_arena" ]]; then
  echo "--full-pipeline is only valid with HUGE_STAGE3_DATA=humanoid_arena" >&2
  exit 2
fi

validate_humanoid_arena_matrix() {
  if [[ "${#HUMANOID_ARENA_ARCHITECTURES[@]}" -eq 0 ]]; then
    echo "HumanoidArena architecture list must not be empty" >&2
    return 2
  fi
  if [[ "${#HUMANOID_ARENA_TUNING_MODES[@]}" -eq 0 ]]; then
    echo "HumanoidArena tuning mode list must not be empty" >&2
    return 2
  fi
  local architecture
  for architecture in "${HUMANOID_ARENA_ARCHITECTURES[@]}"; do
    case "$architecture" in
      qwenpi_v3|qwengr00t|qwenoft) ;;
      *) echo "invalid HumanoidArena architecture: $architecture" >&2; return 2 ;;
    esac
  done
  local tuning_mode
  for tuning_mode in "${HUMANOID_ARENA_TUNING_MODES[@]}"; do
    case "$tuning_mode" in
      full|lora|frozen_vlm) ;;
      *) echo "invalid HumanoidArena tuning mode: $tuning_mode" >&2; return 2 ;;
    esac
  done
  if [[ -n "$HUMANOID_ARENA_RESUME_STATE" ]]; then
    if [[ "${#HUMANOID_ARENA_ARCHITECTURES[@]}" -ne 1 ]] ||
       [[ "${#HUMANOID_ARENA_TUNING_MODES[@]}" -ne 1 ]] ||
       [[ "${HUMANOID_ARENA_TUNING_MODES[0]}" != "full" ]]; then
      echo "HUGE_HUMANOID_ARENA_RESUME_STATE requires exactly one architecture in full mode" >&2
      return 2
    fi
    case "${HUMANOID_ARENA_INIT_CKPT,,}" in
      ""|none|null) ;;
      *) echo "HumanoidArena resume cannot also set HUGE_HUMANOID_ARENA_INIT_CKPT" >&2; return 2 ;;
    esac
  fi
  if truthy "$FULL_PIPELINE"; then
    if [[ "${#HUMANOID_ARENA_TUNING_MODES[@]}" -ne 1 ]] ||
       [[ "${HUMANOID_ARENA_TUNING_MODES[0]}" != "full" ]]; then
      echo "--full-pipeline requires exactly the full tuning mode" >&2
      return 2
    fi
    if [[ -n "$HUMANOID_ARENA_RESUME_STATE" ]]; then
      echo "--full-pipeline cannot resume an existing Stage 3 run" >&2
      return 2
    fi
    case "${HUMANOID_ARENA_INIT_CKPT,,}" in
      ""|none|null) ;;
      *) echo "--full-pipeline cannot set HUGE_HUMANOID_ARENA_INIT_CKPT" >&2; return 2 ;;
    esac
    if [[ ! -d "$STAGE12_DATA_ROOT" ]]; then
      echo "Stage 1/2 dataset root not found: $STAGE12_DATA_ROOT" >&2
      return 2
    fi
    if [[ ! -f "$STAGE2_PROPRIO_STATS" ]]; then
      echo "Stage 2 proprio stats not found: $STAGE2_PROPRIO_STATS" >&2
      return 2
    fi
  fi
  if [[ " ${HUMANOID_ARENA_TUNING_MODES[*]} " == *" full "* ]]; then
    if [[ -n "${HUGE_NUM_PROCESSES:-}" && "${HUGE_NUM_PROCESSES}" != "8" ]]; then
      echo "HumanoidArena G1 full requires HUGE_NUM_PROCESSES=8 when set" >&2
      return 2
    fi
    if [[ -n "${HUGE_EXPECTED_NUM_PROCESSES:-}" && "${HUGE_EXPECTED_NUM_PROCESSES}" != "8" ]]; then
      echo "HumanoidArena G1 full requires HUGE_EXPECTED_NUM_PROCESSES=8 when set" >&2
      return 2
    fi
  fi
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

  echo "Flag file already exists, another run_huge.sh is running: $LOCK_FILE" >&2
  if [[ -r "$LOCK_FILE" ]]; then
    echo "Existing flag:" >&2
    sed 's/^/  /' "$LOCK_FILE" >&2 || true
  fi
  exit 2
}

release_run_lock() {
  if [[ -f "$LOCK_FILE" ]] && grep -qx "token=$LOCK_TOKEN" "$LOCK_FILE" 2>/dev/null; then
    rm -f "$LOCK_FILE"
  fi
}

run_exp() {
  local stage="$1"
  shift

  local out_dir="$RUN_ROOT/$EXPERIMENT_NAME"
  local cmd=("$TRAIN_SCRIPT" "$stage" --output-dir "$out_dir")

  cmd+=("$@")

  RUN_COUNT=$((RUN_COUNT + 1))
  echo
  echo "==> [$RUN_COUNT] $EXPERIMENT_NAME/$stage"
  printf '==> command:'
  printf ' %q' "${cmd[@]}"
  echo

  if truthy "$DRY_RUN"; then
    return 0
  fi

  "${cmd[@]}"
}

run_stage1() {
  local sample_stride="$1"
  run_exp stage1 --model-architecture "$MODEL_ARCHITECTURE" --sample-stride "$sample_stride"
}

run_stage2() {
  run_exp stage2 --model-architecture "$MODEL_ARCHITECTURE"
}

run_sim_stage3() {
  run_exp stage3 \
    --model-architecture "$MODEL_ARCHITECTURE" \
    --prop-roots "$SIM_ROOT" \
    --proprio-stats-path "$SIM_STATS" \
    --epochs "$SIM_STAGE3_EPOCHS" \
    --checkpointing-steps "$SIM_STAGE3_CHECKPOINTING_STEPS"
}

run_humanoid_arena_stage1() {
  local architecture="$1"
  run_exp stage1 \
    --model-architecture "$architecture" \
    --dataset-root "$STAGE12_DATA_ROOT" \
    --vlm-learning-rate "$STAGE12_VLM_LR" \
    --projector-learning-rate "$STAGE12_PROJECTOR_LR" \
    --action-learning-rate "$STAGE12_ACTION_LR" \
    --num-gpus 8 \
    --num-nodes 1 \
    --mixed-precision bf16
}

run_humanoid_arena_stage2() {
  local architecture="$1"
  run_exp stage2 \
    --model-architecture "$architecture" \
    --dataset-root "$STAGE12_DATA_ROOT" \
    --proprio-stats-path "$STAGE2_PROPRIO_STATS" \
    --vlm-learning-rate "$STAGE12_VLM_LR" \
    --projector-learning-rate "$STAGE12_PROJECTOR_LR" \
    --action-learning-rate "$STAGE12_ACTION_LR" \
    --num-gpus 8 \
    --num-nodes 1 \
    --mixed-precision bf16
}

# Resolve the latest stage2 checkpoint for a stride under STAGE2_SOURCE_ROOT.
# Mirrors resolve_latest_stage_ckpt in train_huge.sh but reads the source root.
resolve_stage2_ckpt() {
  local sample_stride="$1"
  local stage2_root="$STAGE2_SOURCE_ROOT/stride_${sample_stride}/stage2"
  [[ -d "$stage2_root" ]] || return 1
  local latest_run
  latest_run="$(find "$stage2_root" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -1)"
  [[ -n "$latest_run" ]] || return 1
  local latest_ckpt
  latest_ckpt="$(find "$latest_run" -mindepth 1 -maxdepth 1 -type d -name 'step_*' 2>/dev/null | sort -V | tail -1)"
  [[ -n "$latest_ckpt" ]] || return 1
  echo "$latest_ckpt"
}

run_real_stage3() {
  local sample_stride="$1"
  local init_ckpt
  if ! init_ckpt="$(resolve_stage2_ckpt "$sample_stride")"; then
    echo "==> ERROR: no stage2 checkpoint for stride_${sample_stride} under $STAGE2_SOURCE_ROOT" >&2
    echo "==>        expected $STAGE2_SOURCE_ROOT/stride_${sample_stride}/stage2/<run>/step_*" >&2
    exit 1
  fi
  run_exp stage3 \
    --model-architecture "$MODEL_ARCHITECTURE" \
    --init-ckpt "$init_ckpt" \
    --prop-roots "$REAL_ROOT" \
    --proprio-stats-path "$REAL_STATS" \
    --epochs "$REAL_STAGE3_EPOCHS" \
    --checkpointing-steps "$REAL_STAGE3_CHECKPOINTING_STEPS"
}

# Stage 3 on the unified HumanoidArena cache, optionally filtered to one task.
run_humanoid_arena_stage3() {
  local architecture="$1"
  local tuning_mode="$2"
  local tuning_args=()
  local initialization_args=()
  local recipe_args=()
  if [[ -n "$HUMANOID_ARENA_RESUME_STATE" ]]; then
    initialization_args=(--resume-from-checkpoint "$HUMANOID_ARENA_RESUME_STATE")
  elif ! truthy "$FULL_PIPELINE"; then
    initialization_args=(--init-ckpt "$HUMANOID_ARENA_INIT_CKPT")
  fi
  case "$tuning_mode" in
    full)
      tuning_args=(--no-freeze-qwen --no-qwen-lora --no-train-qwen-lora)
      local max_steps
      case "$architecture" in
        qwenoft) max_steps="$HUMANOID_ARENA_QWENOFT_MAX_STEPS" ;;
        qwenpi_v3) max_steps="$HUMANOID_ARENA_QWENPI_V3_MAX_STEPS" ;;
        qwengr00t) max_steps="$HUMANOID_ARENA_QWENGR00T_MAX_STEPS" ;;
      esac
      recipe_args=(
        --num-gpus 8
        --num-nodes 1
        --mixed-precision bf16
        --vlm-learning-rate "$HUMANOID_ARENA_VLM_LR"
        --projector-learning-rate "$HUMANOID_ARENA_PROJECTOR_LR"
        --action-learning-rate "$HUMANOID_ARENA_ACTION_LR"
        --weight-decay 0.005
        --lr-scheduler-type cosine
        --warmup-ratio "$HUMANOID_ARENA_WARMUP_RATIO"
        --min-lr-rate "$HUMANOID_ARENA_MIN_LR_RATE"
        --max-steps "$max_steps"
        --gradient-accumulation-steps 1
        --checkpointing-steps "$HUMANOID_ARENA_CHECKPOINTING_STEPS"
        --resume-state-steps "$HUMANOID_ARENA_CHECKPOINTING_STEPS"
        --resume-state-keep "$HUMANOID_ARENA_RESUME_STATE_KEEP"
      )
      ;;
    lora)
      tuning_args=(
        --no-freeze-qwen
        --qwen-lora
        --train-qwen-lora
        --qwen-lora-r "$HUMANOID_ARENA_LORA_R"
        --qwen-lora-alpha "$HUMANOID_ARENA_LORA_ALPHA"
        --qwen-lora-dropout "$HUMANOID_ARENA_LORA_DROPOUT"
      )
      ;;
    frozen_vlm)
      tuning_args=(--freeze-qwen --no-qwen-lora --no-train-qwen-lora)
      ;;
  esac
  if [[ "$tuning_mode" != "full" ]]; then
    recipe_args=(
      --learning-rate "$HUMANOID_ARENA_LEGACY_LR"
      --epochs "$HUMANOID_ARENA_STAGE3_EPOCHS"
      --checkpointing-steps "$HUMANOID_ARENA_CHECKPOINTING_STEPS"
    )
  fi
  run_exp stage3 \
    --model-architecture "$architecture" \
    "${initialization_args[@]}" \
    --prop-roots \
    --humanoid-arena-roots "$HUMANOID_ARENA_ROOT" \
    --humanoid-arena-task "$HUMANOID_ARENA_TASK" \
    --proprio-stats-path "$HUMANOID_ARENA_STATS" \
    "${recipe_args[@]}" \
    "${tuning_args[@]}"
}

run_all() {
  if [[ "$STAGE3_DATA" == "humanoid_arena" ]]; then
    validate_humanoid_arena_matrix
    local architecture
    local tuning_mode
    for architecture in "${HUMANOID_ARENA_ARCHITECTURES[@]}"; do
      for tuning_mode in "${HUMANOID_ARENA_TUNING_MODES[@]}"; do
        EXPERIMENT_NAME="$HUMANOID_ARENA_TASK/$architecture/$tuning_mode"
        if truthy "$FULL_PIPELINE"; then
          run_humanoid_arena_stage1 "$architecture"
          run_humanoid_arena_stage2 "$architecture"
        fi
        run_humanoid_arena_stage3 "$architecture" "$tuning_mode"
      done
    done
    return
  fi
  local sample_stride
  for sample_stride in "${STAGE1_SAMPLE_STRIDES[@]}"; do
    EXPERIMENT_NAME="stride_${sample_stride}"
    if [[ "$STAGE3_DATA" == "real" ]]; then
      run_real_stage3 "$sample_stride"
    else
      run_stage1 "$sample_stride"
      run_stage2
      run_sim_stage3
    fi
  done
}

acquire_run_lock
run_all

if [[ "$RUN_COUNT" -eq 0 ]]; then
  cat <<EOF
No experiments are enabled.

Edit scripts/train/run_huge.sh and add run_exp blocks inside run_all().
Use HUGE_DRY_RUN=1 scripts/train/run_huge.sh to preview commands.
EOF
fi
