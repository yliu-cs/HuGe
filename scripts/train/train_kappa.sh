#!/usr/bin/env bash
# Kappa full training pipeline.
#
# Runs the four stages sequentially and threads each stage's output run
# directory into the next stage's --train.pretrained-weights, so the chain
# pretrain(egodex) -> pretrain(he_wbt) -> posttrain -> finetune
# starts each stage from the previous stage's last saved checkpoint without
# any hand-edited paths.
#
# Usage:
#   ./scripts/train/run_kappa_pipeline.sh                                  # full chain
#   ./scripts/train/run_kappa_pipeline.sh --num_gpus 8                     # forward to all stages
#   ./scripts/train/run_kappa_pipeline.sh --from posttrain \
#       --pretrained-weights .runs/kappa/<existing-stage1-2-run>           # resume mid-pipeline
#
# Anything not consumed by this script is forwarded verbatim to every
# sub-stage (e.g. --num_gpus, --no_deepspeed, --master_port, --data.* /
# --train.* / --model.* config overrides). Per-stage overrides aren't
# supported here; if you need them, run the stage scripts directly.
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

RUNS_DIR=".runs/kappa"

START_STAGE="egodex"
SEED_WEIGHTS=""
FORWARDED=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --from)
            START_STAGE="$2"; shift 2 ;;
        --pretrained-weights|--train.pretrained-weights)
            SEED_WEIGHTS="$2"; shift 2 ;;
        --train.pretrained-weights=*)
            SEED_WEIGHTS="${1#*=}"; shift ;;
        --pretrained-weights=*)
            SEED_WEIGHTS="${1#*=}"; shift ;;
        *)
            FORWARDED+=("$1"); shift ;;
    esac
done

STAGES=(egodex he_wbt posttrain finetune)
START_IDX=-1
for i in "${!STAGES[@]}"; do
    if [ "${STAGES[$i]}" = "$START_STAGE" ]; then
        START_IDX=$i
        break
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
    extra=()
    if [ -n "$CURRENT_WEIGHTS" ]; then
        extra+=(--train.pretrained-weights "$CURRENT_WEIGHTS")
    fi
    case "$stage" in
        egodex|he_wbt)
            run_stage "$stage" \
                "$SCRIPT_DIR/pretrain_kappa.sh" --stage "$stage" \
                "${extra[@]}" "${FORWARDED[@]}"
            ;;
        posttrain)
            run_stage "$stage" \
                "$SCRIPT_DIR/posttrain_kappa.sh" \
                "${extra[@]}" "${FORWARDED[@]}"
            ;;
        finetune)
            run_stage "$stage" \
                "$SCRIPT_DIR/finetune_kappa.sh" \
                "${extra[@]}" "${FORWARDED[@]}"
            ;;
    esac
    CURRENT_WEIGHTS="$LATEST_RUN"
done

echo "============================================================"
echo "==> [pipeline] all stages completed"
echo "==> [pipeline] final run: $CURRENT_WEIGHTS"
echo "============================================================"
