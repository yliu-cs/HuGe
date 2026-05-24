#!/usr/bin/env bash
# Sequential data-ablation sweep for Kappa fine-tuning.
#
# Runs 13 fine-tune tasks back-to-back, all seeded from the same Post-Train
# checkpoint (.runs/kappa/exp.kappa.b1024.gpus8.2605200648). For each task we
# pick N random *episodes* from real/2026-05-23 and (optionally) M random
# episodes from ego/2026-5-23 via PropDataConfig.max_samples and
# PropEgoDataConfig.max_samples.
#
# Naming convention: each task is identified by "<R>R" or "<R>R+<E>E" where
# the numbers count episodes. For mixed tasks we run ego first (no proprio,
# no RTC — like Pre-Train EgoDex) and then load that checkpoint to run real
# (with proprio + RTC — like Post-Train), mirroring the PreTrain -> PostTrain
# bridge that the base checkpoint itself was built with.
#
# Step budget: each task gets TOTAL_STEPS (2000) steps in aggregate, split
# between ego and real in proportion to their episode counts and rounded to
# the nearest STEP_ROUND (100). Pure-R tasks spend the full budget on real.
#
# Output layout: each task drops its actual run dirs under .runs/kappa/ with
# auto-generated names (<exp>.kappa.b<BS>.gpus<N>.<TS>/) per the trainer's
# convention; we then create a friendly directory ".runs/kappa/<task>/" with
# symlinks "ego/" and/or "real/" pointing to the stage runs.
#
# Usage:
#   ./scripts/train/sweep_data_ablation.sh                # full sweep
#   ./scripts/train/sweep_data_ablation.sh --from 50R+50E # resume at a task
#   ./scripts/train/sweep_data_ablation.sh --only 100R    # one task only
#
# Forwarded flags (anything not consumed here) go to every train_kappa.sh
# invocation — e.g. --num_gpus 8 / --master_port 29501 / --zero_stage 2.
#
# Do NOT run other Kappa training jobs in parallel while this sweep is
# active; the marker-file discovery that locates each stage's auto-named
# run dir will get confused.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

LAUNCHER="$SCRIPT_DIR/train_kappa.sh"
RUNS_DIR=".runs/kappa"
BASE_CKPT=".runs/kappa/exp.kappa.b1024.gpus8.2605200648"

REAL_ROOT="resource/dataset/proprietary/real/2026-05-23"
EGO_ROOT="resource/dataset/proprietary/ego/2026-5-23"

REAL_CONFIG="finetune_prop_ep_kappa_config"
EGO_CONFIG="finetune_prop_ego_kappa_config"

# Per-task total step budget and rounding granularity for the ego/real split.
TOTAL_STEPS=2000
STEP_ROUND=100

# 13 tasks, ordered cheapest first so the sweep makes progress before the
# heaviest jobs. Each entry is "<task_name>:<R_episodes>:<E_episodes>".
TASKS=(
    "20R:20:0"
    "50R:50:0"
    "100R:100:0"
    "170R:170:0"
    "20R+10E:20:10"
    "20R+20E:20:20"
    "20R+40E:20:40"
    "50R+25E:50:25"
    "50R+50E:50:50"
    "50R+100E:50:100"
    "100R+50E:100:50"
    "100R+100E:100:100"
    "170R+85E:170:85"
)

START_TASK=""
ONLY_TASK=""
FORWARDED=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --from) START_TASK="$2"; shift 2 ;;
        --only) ONLY_TASK="$2"; shift 2 ;;
        *)      FORWARDED+=("$1"); shift ;;
    esac
done

if [ ! -d "$BASE_CKPT" ]; then
    echo "Error: BASE_CKPT not found: $BASE_CKPT" >&2
    exit 1
fi
mkdir -p "$RUNS_DIR"

discover_latest_run() {
    # Pick the most recently modified top-level run dir under $RUNS_DIR
    # whose mtime is strictly newer than the marker file. Symlinks (our own
    # <task>/ friendly dirs) are excluded so they don't shadow real runs.
    local marker="$1"
    find "$RUNS_DIR" -mindepth 1 -maxdepth 1 -type d -newer "$marker" \
        -printf '%T@ %p\n' 2>/dev/null \
        | sort -rn | head -1 | awk '{print $2}'
}

run_stage() {
    # Args: <label> <command...>
    # Returns the produced run dir in $LATEST_RUN; aborts on stage failure
    # or if no new run dir appears under $RUNS_DIR.
    local label="$1"; shift
    local marker
    marker=$(mktemp -t "kappa_sweep_${label//[^A-Za-z0-9]/_}.XXXXXX.mark")
    sleep 1  # 1 s mtime granularity guard

    echo "============================================================"
    echo "==> [sweep] stage: $label"
    echo "==> [sweep] cmd: $*"
    echo "============================================================"

    set +e
    "$@"
    local rc=$?
    set -e

    if [ "$rc" -ne 0 ]; then
        rm -f "$marker"
        echo "==> [sweep] stage $label FAILED (rc=$rc) — aborting" >&2
        exit "$rc"
    fi

    LATEST_RUN=$(discover_latest_run "$marker")
    rm -f "$marker"

    if [ -z "$LATEST_RUN" ]; then
        echo "==> [sweep] stage $label produced no new run dir under $RUNS_DIR" >&2
        exit 1
    fi
    if [ ! -d "$LATEST_RUN/checkpoints" ]; then
        echo "==> [sweep] stage $label run $LATEST_RUN has no checkpoints/ subdir" >&2
        exit 1
    fi
    echo "==> [sweep] stage $label -> $LATEST_RUN"
}

link_stage() {
    # Args: <task_dir> <stage_name> <run_dir>
    # Re-link <task_dir>/<stage_name> -> <run_dir> as a relative symlink so
    # the friendly directory survives moves of .runs/kappa.
    local task_dir="$1" stage_name="$2" run_dir="$3"
    mkdir -p "$task_dir"
    local rel
    rel=$(realpath --relative-to="$task_dir" "$run_dir")
    ln -sfn "$rel" "$task_dir/$stage_name"
}

split_steps() {
    # Echo "<ego_steps> <real_steps>" for a (R, E) split, summing to
    # TOTAL_STEPS and snapped to STEP_ROUND. Pure-R tasks (E=0) put the
    # full budget on real; pure-E tasks (R=0) put it on ego.
    local R="$1" E="$2"
    local ego_steps real_steps
    if [ "$E" -eq 0 ]; then
        ego_steps=0
        real_steps=$TOTAL_STEPS
    elif [ "$R" -eq 0 ]; then
        ego_steps=$TOTAL_STEPS
        real_steps=0
    else
        local total=$((R + E))
        # round(E/total * TOTAL_STEPS / STEP_ROUND) * STEP_ROUND, integer-only.
        local units=$((TOTAL_STEPS / STEP_ROUND))      # = 20 for 2000/100
        ego_steps=$(( (2 * E * units + total) / (2 * total) * STEP_ROUND ))
        real_steps=$((TOTAL_STEPS - ego_steps))
    fi
    echo "$ego_steps $real_steps"
}

run_task() {
    # Args: <task_name> <R_count> <E_count>
    local task="$1" R="$2" E="$3"
    local task_dir="$RUNS_DIR/$task"

    local ego_steps real_steps
    read -r ego_steps real_steps <<<"$(split_steps "$R" "$E")"

    echo
    echo "############################################################"
    echo "## task: $task   (R=$R, E=$E)   steps: ego=$ego_steps real=$real_steps"
    echo "############################################################"

    local real_seed_ckpt="$BASE_CKPT"

    if [ "$E" -gt 0 ]; then
        # Stage 1: ego (no proprio / no RTC), seeded from BASE_CKPT.
        run_stage "${task}.ego" \
            "$LAUNCHER" "$EGO_CONFIG" \
            --exp "${task}_ego" \
            --train.pretrained-weights "$BASE_CKPT" \
            --train.max-training-steps "$ego_steps" \
            --data.data-roots "$EGO_ROOT" \
            --data.max-samples "$E" \
            "${FORWARDED[@]}"
        local ego_run="$LATEST_RUN"
        link_stage "$task_dir" "ego" "$ego_run"
        # Stage 2 will load this ego run as its pretrained-weights so the
        # real fine-tune picks up the ego-distilled features.
        real_seed_ckpt="$ego_run"
    fi

    # Stage 2 (or only stage, for R-only tasks): real (proprio + RTC).
    run_stage "${task}.real" \
        "$LAUNCHER" "$REAL_CONFIG" \
        --exp "${task}_real" \
        --train.pretrained-weights "$real_seed_ckpt" \
        --train.max-training-steps "$real_steps" \
        --data.data-roots "$REAL_ROOT" \
        --data.max-samples "$R" \
        "${FORWARDED[@]}"
    link_stage "$task_dir" "real" "$LATEST_RUN"

    echo "==> [sweep] task $task complete: $task_dir"
}

# Resolve --from / --only into the slice of tasks to run.
SELECTED=()
if [ -n "$ONLY_TASK" ]; then
    for entry in "${TASKS[@]}"; do
        IFS=':' read -r name _ _ <<<"$entry"
        if [ "$name" = "$ONLY_TASK" ]; then
            SELECTED+=("$entry")
        fi
    done
    if [ "${#SELECTED[@]}" -eq 0 ]; then
        echo "Error: --only '$ONLY_TASK' not in task list" >&2
        exit 1
    fi
else
    start_idx=0
    if [ -n "$START_TASK" ]; then
        start_idx=-1
        for i in "${!TASKS[@]}"; do
            IFS=':' read -r name _ _ <<<"${TASKS[$i]}"
            if [ "$name" = "$START_TASK" ]; then
                start_idx=$i; break
            fi
        done
        if [ "$start_idx" -lt 0 ]; then
            echo "Error: --from '$START_TASK' not in task list" >&2
            exit 1
        fi
    fi
    for ((i = start_idx; i < ${#TASKS[@]}; i++)); do
        SELECTED+=("${TASKS[$i]}")
    done
fi

echo "==> [sweep] base checkpoint: $BASE_CKPT"
echo "==> [sweep] step budget per task: $TOTAL_STEPS (rounded to $STEP_ROUND)"
echo "==> [sweep] running ${#SELECTED[@]} task(s):"
for entry in "${SELECTED[@]}"; do
    IFS=':' read -r name R E <<<"$entry"
    read -r es rs <<<"$(split_steps "$R" "$E")"
    echo "==>          $name (R=$R, E=$E)  steps: ego=$es real=$rs"
done

for entry in "${SELECTED[@]}"; do
    IFS=':' read -r name R E <<<"$entry"
    run_task "$name" "$R" "$E"
done

echo "============================================================"
echo "==> [sweep] all tasks complete; results under $RUNS_DIR/<task>/"
echo "============================================================"
