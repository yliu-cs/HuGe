#!/usr/bin/env bash
set -euo pipefail

# Experiment runner for the default HumanoidArena 3-architecture x 3-tuning-mode
# matrix, with the legacy sim and real pipelines selected by HUGE_STAGE3_DATA.
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
# humanoid_arena mode (the default) trains stage3 from scratch on HOI_pp_box
# once per architecture x tuning mode. It writes to:
#   ${HUGE_RUN_ROOT:-./.runs/stage3_humanoidarena}/
#     HOI_pp_box/<arch>/<full|lora|frozen_vlm>/stage3/<ts>/
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
#   HUGE_HUMANOID_ARENA_TASK=...   exact HumanoidArena task name, or all (default HOI_pp_box)
#   HUGE_HUMANOID_ARENA_INIT_CKPT=...  optional init checkpoint (default none/from scratch)
#   HUGE_HUMANOID_ARENA_ARCHITECTURES=...     space-separated archs (default "qwenpi_v3 qwengr00t qwenoft")
#   HUGE_HUMANOID_ARENA_TUNING_MODES=...      space-separated modes (default "full lora frozen_vlm")
#   HUGE_HUMANOID_ARENA_LORA_R=...            language-attention LoRA rank (default 8)
#   HUGE_HUMANOID_ARENA_LORA_ALPHA=...        language-attention LoRA alpha (default 16)
#   HUGE_HUMANOID_ARENA_LORA_DROPOUT=...      language-attention LoRA dropout (default 0.0)
#   HUGE_HUMANOID_ARENA_STAGE3_EPOCHS=...      humanoid_arena stage3 epochs (default 50)
#   HUGE_HUMANOID_ARENA_STAGE3_CHECKPOINTING_STEPS=...  humanoid_arena ckpt interval (default 3000)
#   HUGE_MODEL_ARCHITECTURE=...    qwenpi_v3, qwengr00t, or qwenoft (sim/real modes only)
#   HUGE_LOCK_FILE=...             override the default run flag file
#
# Usage:
#   scripts/train/run_huge.sh              run the default HOI_pp_box 3x3 matrix
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
  humanoid_arena) DEFAULT_RUN_ROOT="./.runs/stage3_humanoidarena" ;;
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

# --- humanoid_arena stage3 (from scratch, architecture x tuning-mode matrix) ---
HUMANOID_ARENA_TASK="${HUGE_HUMANOID_ARENA_TASK:-HOI_pp_box}"
HUMANOID_ARENA_ROOT="${HUGE_STAGE3_HUMANOID_ARENA_ROOT:-resource/datasets/HumanoidArena}"
HUMANOID_ARENA_STATS="${HUGE_STAGE3_HUMANOID_ARENA_PROPRIO_STATS:-$HUMANOID_ARENA_ROOT/meta/proprio_stats_union_sonic.json}"
HUMANOID_ARENA_INIT_CKPT="${HUGE_HUMANOID_ARENA_INIT_CKPT:-none}"
HUMANOID_ARENA_STAGE3_EPOCHS="${HUGE_HUMANOID_ARENA_STAGE3_EPOCHS:-50}"
HUMANOID_ARENA_STAGE3_CHECKPOINTING_STEPS="${HUGE_HUMANOID_ARENA_STAGE3_CHECKPOINTING_STEPS:-3000}"
read -r -a HUMANOID_ARENA_ARCHITECTURES <<< "${HUGE_HUMANOID_ARENA_ARCHITECTURES:-qwenpi_v3 qwengr00t qwenoft}"
read -r -a HUMANOID_ARENA_TUNING_MODES <<< "${HUGE_HUMANOID_ARENA_TUNING_MODES:-full lora frozen_vlm}"
HUMANOID_ARENA_LORA_R="${HUGE_HUMANOID_ARENA_LORA_R:-8}"
HUMANOID_ARENA_LORA_ALPHA="${HUGE_HUMANOID_ARENA_LORA_ALPHA:-16}"
HUMANOID_ARENA_LORA_DROPOUT="${HUGE_HUMANOID_ARENA_LORA_DROPOUT:-0.0}"

read -r -a STAGE1_SAMPLE_STRIDES <<< "${HUGE_STAGE1_SAMPLE_STRIDES:-1 2 5 10 50}"
SELECTED_STRIDE=""

usage() {
  cat <<EOF
Usage: $0 [--stride STRIDE] [STRIDE]

By default, run the HumanoidArena HOI_pp_box architecture x tuning-mode matrix.
With HUGE_STAGE3_DATA=sim|real, optionally select one stride; configured strides:
${STAGE1_SAMPLE_STRIDES[*]}
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
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

LOCK_SCOPE="all"
if [[ "$STAGE3_DATA" == "humanoid_arena" ]]; then
  LOCK_SCOPE="humanoid_arena_${HUMANOID_ARENA_TASK}"
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
  exit 0
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
  case "$tuning_mode" in
    full)
      tuning_args=(--no-freeze-qwen --no-qwen-lora --no-train-qwen-lora)
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
  run_exp stage3 \
    --model-architecture "$architecture" \
    --init-ckpt "$HUMANOID_ARENA_INIT_CKPT" \
    --prop-roots \
    --humanoid-arena-roots "$HUMANOID_ARENA_ROOT" \
    --humanoid-arena-task "$HUMANOID_ARENA_TASK" \
    --proprio-stats-path "$HUMANOID_ARENA_STATS" \
    --epochs "$HUMANOID_ARENA_STAGE3_EPOCHS" \
    --checkpointing-steps "$HUMANOID_ARENA_STAGE3_CHECKPOINTING_STEPS" \
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
