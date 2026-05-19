#!/usr/bin/env bash
# One-command launcher for the Kappa real-robot deploy stack.
#
# Spins up four tmux windows in a session named `kappa_deploy`:
#   1. serve  — Kappa WebSocket inference server (src.inference.kappa_serve_rtc)
#   2. wbc    — WBC C++ deploy (gear_sonic_deploy/deploy.sh --input-type zmq_manager real)
#   3. bridge — Kappa <-> WBC bridge (scripts/deploy/run_kappa_inference.py)
#   4. kbd    — Keyboard publisher (scripts/deploy/keyboard_pub.py)
#
# Each window is wrapped with `script -q -c "..." <log>` so the pane output is
# mirrored to .runs/deploy/<timestamp>/<window>.log for later inspection.
#
# Required:
#   KAPPA_CKPT — absolute or repo-relative path to the Fine-Tune model.safetensors
# Override any other config via the env vars listed below.

set -euo pipefail

cd "$(dirname "$0")/../.."   # cd to Kappa repo root

# ---------- env defaults (override before invocation) ----------
: "${KAPPA_CKPT:?must set KAPPA_CKPT to Fine-Tune model.safetensors path}"
: "${KAPPA_BACKBONE:=resource/model/Qwen3-VL-2B-Instruct}"
: "${KAPPA_PROPRIO_STATS:=resource/dataset/G1_WBT-cache/meta/g1_proprio_stats.json}"
: "${KAPPA_HOST:=0.0.0.0}"
: "${WS_PORT:=8765}"
: "${CTRL_HZ:=50}"
: "${KAPPA_DEVICE:=cuda:0}"

: "${WBC_MODE:=real}"               # `real` for hardware, `sim` for MuJoCo loopback
: "${WBC_INPUT_TYPE:=zmq_manager}"

: "${WS_CLIENT_HOST:=localhost}"    # how the bridge reaches the server
: "${CAMERA_HOST:=localhost}"
: "${CAMERA_PORT:=5555}"
: "${STATE_HOST:=localhost}"
: "${STATE_PORT:=5557}"
: "${ACTION_HOST:=0.0.0.0}"
: "${ACTION_PORT:=5556}"
: "${KEYBOARD_HOST:=localhost}"
: "${KEYBOARD_PORT:=5580}"
: "${BRIDGE_PROMPT:=perform the task}"

: "${SESSION:=kappa_deploy}"

# ---------- session guard ----------
if tmux has-session -t "$SESSION" 2>/dev/null; then
    cat >&2 <<EOF
[deploy] tmux session '$SESSION' already exists.
         attach:  tmux attach -t $SESSION
         restart: tmux kill-session -t $SESSION && $0
EOF
    exit 1
fi

if [[ ! -x .venv/bin/python ]]; then
    echo "[deploy] .venv/bin/python not found. Run \`uv sync\` first." >&2
    exit 1
fi
if [[ ! -x GR00T-WholeBodyControl/gear_sonic_deploy/deploy.sh ]]; then
    echo "[deploy] WBC deploy.sh not found / not executable." >&2
    exit 1
fi

LOGDIR=.runs/deploy/$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOGDIR"
echo "[deploy] logs -> $LOGDIR"

PY=.venv/bin/python

SERVE_CMD="$PY -m src.inference.kappa_serve_rtc \
  --weights '$KAPPA_CKPT' \
  --backbone '$KAPPA_BACKBONE' \
  --proprio-stats '$KAPPA_PROPRIO_STATS' \
  --normalize-mode bounds_q99 \
  --device '$KAPPA_DEVICE' \
  --ctrl-hz $CTRL_HZ \
  --host $KAPPA_HOST --port $WS_PORT"

WBC_CMD="cd GR00T-WholeBodyControl/gear_sonic_deploy && ./deploy.sh --input-type $WBC_INPUT_TYPE $WBC_MODE"

BRIDGE_CMD="$PY scripts/deploy/run_kappa_inference.py \
  --ws-host $WS_CLIENT_HOST --ws-port $WS_PORT \
  --camera-host $CAMERA_HOST --camera-port $CAMERA_PORT \
  --state-zmq-host $STATE_HOST --state-zmq-port $STATE_PORT \
  --action-zmq-host $ACTION_HOST --action-zmq-port $ACTION_PORT \
  --keyboard-zmq-host $KEYBOARD_HOST --keyboard-zmq-port $KEYBOARD_PORT \
  --prompt '$BRIDGE_PROMPT' \
  --tick-hz $CTRL_HZ"

KBD_CMD="$PY scripts/deploy/keyboard_pub.py --port $KEYBOARD_PORT"

# ---------- spawn ----------
# Server first; it takes longest to come up (model load + cuDNN warmup).
tmux new-session -d -s "$SESSION" -n serve \
    "script -q -c \"$SERVE_CMD\" $LOGDIR/serve.log"

# Brief pause so the next windows don't race the server's port bind log.
sleep 0.3

tmux new-window -t "$SESSION" -n wbc \
    "script -q -c \"$WBC_CMD\" $LOGDIR/wbc.log"
tmux new-window -t "$SESSION" -n bridge \
    "script -q -c \"$BRIDGE_CMD\" $LOGDIR/bridge.log"
tmux new-window -t "$SESSION" -n kbd \
    "script -q -c \"$KBD_CMD\" $LOGDIR/kbd.log"

cat <<EOF
[deploy] session '$SESSION' launched.
         attach:        tmux attach -t $SESSION
         next window:   Ctrl-b n     prev: Ctrl-b p
         detach:        Ctrl-b d
         stop all:      tmux kill-session -t $SESSION
         logs:          $LOGDIR/{serve,wbc,bridge,kbd}.log
EOF
