#!/usr/bin/env bash
# Kappa full training pipeline.
#
# Runs the three stages sequentially and threads each stage's output run
# directory into the next stage's --train.pretrained-weights, so the chain
# pretrain(egodex) -> posttrain (HE+WBT, SONIC) -> finetune (proprietary, SONIC)
# starts each stage from the previous stage's last saved checkpoint without
# any hand-edited paths.
#
# Usage:
#   ./scripts/train/run_kappa.sh                                           # full chain
#   ./scripts/train/run_kappa.sh --num_gpus 8                              # forward to all stages
#   ./scripts/train/run_kappa.sh --from posttrain \
#       --pretrained-weights .runs/kappa/<existing-egodex-run>             # resume mid-pipeline
#
# Anything not consumed by this script is forwarded verbatim to every
# sub-stage (e.g. --num_gpus, --no_deepspeed, --master_port, --data.* /
# --train.* / --model.* config overrides). Per-stage overrides aren't
# supported here; if you need them, run train_kappa.sh directly.
#
# Discovery details:
#   - The trainer writes runs to .runs/<train_cfg.name>/<run_name>/ where
#     train_cfg.name="kappa" for every stage in this project.
#   - Before each stage we drop a marker file; after the stage we pick the
#     most recently modified directory under .runs/kappa/ that's newer than
#     the marker. The trainer's load_pretrained_weights() resolves a run
#     directory to its latest checkpoints/ckpt_*/, so we hand it the run
#     directory instead of a specific checkpoint.
#   - Don't run other Kappa training jobs in parallel during the pipeline,
#     or the discovery may pick the wrong run.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

LAUNCHER="$SCRIPT_DIR/train_kappa.sh"
RUNS_DIR=".runs/kappa"

# Stage -> config name. Adjust this map to switch a stage to a different
# config (e.g. swap finetune to proprietary by pointing it at
# finetune_prop_ep_kappa_config).
declare -A STAGE_CONFIG=(
    [egodex]=pretrain_egodex_kappa_config
    [posttrain]=posttrain_he_wbt_kappa_config
    [finetune]=finetune_prop_ep_kappa_config
)
STAGES=(egodex posttrain finetune)

START_STAGE="egodex"
SEED_WEIGHTS=""
FORWARDED=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --from)
            START_STAGE="$2"; shift 2 ;;
        --pretrained-weights|--train.pretrained-weights)
            SEED_WEIGHTS="$2"; shift 2 ;;
        --train.pretrained-weights=*|--pretrained-weights=*)
            SEED_WEIGHTS="${1#*=}"; shift ;;
        *)
            FORWARDED+=("$1"); shift ;;
    esac
done

START_IDX=-1
for i in "${!STAGES[@]}"; do
    if [ "${STAGES[$i]}" = "$START_STAGE" ]; then
        START_IDX=$i; break
    fi
done
if [ "$START_IDX" -lt 0 ]; then
    echo "Error: --from must be one of: ${STAGES[*]} (got '$START_STAGE')" >&2
    exit 1
fi

if [ "$START_IDX" -gt 0 ] && [ -z "$SEED_WEIGHTS" ]; then
    echo "Error: --from $START_STAGE requires --pretrained-weights to seed the resumed stage" >&2
    exit 1
fi

mkdir -p "$RUNS_DIR"

# Cluster single-runner lock. The user排 N 个八卡节点；only the first node to
# pass this barrier owns the run, the rest detect the flag file and bail so
# the scheduler can reclaim their GPUs. ``set -o noclobber`` + O_EXCL redirect
# is atomic on local FS and on NFS v3+, so the race between concurrent nodes
# is resolved by the kernel/server, not by application code. Flag carries the
# winner's host/pid/start so a stale flag (crashed earlier run) is greppable
# without having to chase the scheduler.
FLAG_FILE=".runs/flag.log"
if ! (set -o noclobber
      printf 'host=%s\npid=%s\nstart=%s\n' \
          "$(hostname)" "$$" "$(date -Iseconds)" > "$FLAG_FILE") 2>/dev/null; then
    echo "==> [pipeline] $FLAG_FILE already exists — another node owns this run:"
    sed 's/^/    /' "$FLAG_FILE" 2>/dev/null || true
    echo "==> [pipeline] exiting cleanly (delete $FLAG_FILE manually to retry)."
    exit 0
fi
echo "==> [pipeline] acquired $FLAG_FILE on $(hostname); other nodes will skip."

discover_latest_run() {
    local marker="$1"
    find "$RUNS_DIR" -mindepth 1 -maxdepth 1 -type d -newer "$marker" \
        -printf '%T@ %p\n' 2>/dev/null \
        | sort -rn | head -1 | awk '{print $2}'
}

# Each stage runs as a subprocess so its `exec` of torchrun/deepspeed
# returns the rc to us. We disable -e around the call to capture rc cleanly.
run_stage() {
    local label="$1"; shift
    local marker
    marker=$(mktemp -t "kappa_pipeline_${label}.XXXXXX.mark")
    # Many filesystems have 1-second mtime granularity; sleep so the marker
    # mtime is strictly older than any directory the stage creates.
    sleep 1

    echo "============================================================"
    echo "==> [pipeline] starting stage: $label"
    echo "==> [pipeline] cmd: $*"
    echo "============================================================"

    set +e
    "$@"
    local rc=$?
    set -e

    if [ "$rc" -ne 0 ]; then
        rm -f "$marker"
        echo "==> [pipeline] stage $label FAILED (rc=$rc) — aborting" >&2
        exit "$rc"
    fi

    LATEST_RUN=$(discover_latest_run "$marker")
    rm -f "$marker"

    if [ -z "$LATEST_RUN" ]; then
        echo "==> [pipeline] stage $label finished but no new run dir under $RUNS_DIR" >&2
        echo "==> [pipeline]   (did the stage save a checkpoint? did another job race us?)" >&2
        exit 1
    fi
    if [ ! -d "$LATEST_RUN/checkpoints" ]; then
        echo "==> [pipeline] stage $label run $LATEST_RUN has no checkpoints/ subdir" >&2
        echo "==> [pipeline]   (likely max_training_steps/checkpointing_steps mismatch)" >&2
        exit 1
    fi

    echo "==> [pipeline] stage $label produced run: $LATEST_RUN"
}

CURRENT_WEIGHTS="$SEED_WEIGHTS"

for ((i = START_IDX; i < ${#STAGES[@]}; i++)); do
    stage="${STAGES[$i]}"
    config="${STAGE_CONFIG[$stage]}"
    extra=()
    [ -n "$CURRENT_WEIGHTS" ] && extra+=(--train.pretrained-weights "$CURRENT_WEIGHTS")
    run_stage "$stage" "$LAUNCHER" "$config" "${extra[@]}" "${FORWARDED[@]}"
    CURRENT_WEIGHTS="$LATEST_RUN"
done

echo "============================================================"
echo "==> [pipeline] all stages completed"
echo "==> [pipeline] final run: $CURRENT_WEIGHTS"
echo "============================================================"

# Release the lock only on full success — ``set -e`` would have exited on any
# stage failure with the flag still in place, which is the desired behavior
# (operator can investigate the crashed run without other nodes stampeding in).
rm -f "$FLAG_FILE"
echo "==> [pipeline] released $FLAG_FILE"
