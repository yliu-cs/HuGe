#!/usr/bin/env bash
# Manual launcher helper for the Kappa real-robot deploy stack.
#
# This script no longer spawns tmux panes or auto-presses keys. It runs the
# pre-flight checks + port reclaim, then prints four commands. The operator
# opens four terminals and runs one command in each:
#   1. serve  — Kappa WebSocket inference server (src.inference.kappa_serve_rtc)
#   2. wbc    — WBC C++ deploy (gear_sonic_deploy/deploy.sh --input-type zmq_manager real)
#   3. bridge — Kappa <-> WBC bridge (scripts/deploy/run_kappa_inference.py)
#   4. kbd    — Keyboard publisher (scripts/deploy/keyboard_pub.py)
#
# All confirmations are manual:
#   y in wbc pane    — confirm "Proceed with deployment? [Y/n]:"
#   k in kbd pane    — start C++ control loop in PLANNER mode (MOTORS ENGAGE)
#   i in kbd pane    — publish LATENT_INITIAL_MOTION_TOKEN + switch to POSE mode
#   p in kbd pane    — unpause the bridge tick loop
#
# Required:
#   KAPPA_CKPT — absolute or repo-relative path to the Fine-Tune model.safetensors
# Override any other config via the env vars listed below.

set -euo pipefail

cd "$(dirname "$0")/../.."   # cd to Kappa repo root

# ---------- env defaults (override before invocation) ----------
: "${KAPPA_CKPT:=.runs/kappa/170R+85E/real.kappa.b1024.gpus8.2605242023/checkpoints/ckpt_3000/model.safetensors}"
: "${KAPPA_BACKBONE:=resource/model/Qwen3-VL-2B-Instruct}"
# Proprio stats are auto-discovered next to KAPPA_CKPT (Trainer bundles
# proprio_stats.json into each ckpt_<step>/). Export KAPPA_PROPRIO_STATS to
# force an override; otherwise leave unset.
: "${KAPPA_PROPRIO_STATS:=}"
: "${KAPPA_HOST:=0.0.0.0}"
: "${WS_PORT:=8765}"
: "${CTRL_HZ:=50}"
: "${KAPPA_DEVICE:=cuda:0}"

: "${WBC_MODE:=real}"               # `real` for hardware, `sim` for MuJoCo loopback
: "${WBC_INPUT_TYPE:=zmq_manager}"
: "${HAND_TYPE:=inspire}"           # `inspire` or `dex3`; MUST match the C++ deploy's --hand-type
                                    # so the bridge thresholds g1_debug.*_q_measured
                                    # using the right polarity (see run_kappa_inference.py).

: "${WS_CLIENT_HOST:=localhost}"    # how the bridge reaches the server
# Real-robot deploy: camera frames are always published by the G1 itself,
# never localhost. Default to the standard Unitree G1 IP; override if your
# robot lives elsewhere on the LAN.
: "${CAMERA_HOST:=192.168.123.164}"
: "${CAMERA_PORT:=5555}"
: "${STATE_HOST:=localhost}"
: "${STATE_PORT:=5557}"
: "${ACTION_HOST:=0.0.0.0}"
: "${ACTION_PORT:=5556}"
: "${KEYBOARD_HOST:=localhost}"
: "${KEYBOARD_PORT:=5580}"
: "${BRIDGE_PROMPT:=Grab the cola, turn left, throw into the trash bin}"

# Per-tick log of (image, proprio, predicted sonic) under this dir, in the
# training-dataset layout. Default: timestamped subdir under .runs/deploy/
# so every real-hardware run is captured automatically. Unset (LOG_DIR=)
# to disable.
: "${LOG_DIR:=.runs/deploy/$(date +%Y%m%d_%H%M%S)}"

# ---------- pre-flight ----------
if [[ ! -x .venv/bin/python ]]; then
    echo "[deploy] .venv/bin/python not found. Run \`uv sync\` first." >&2
    exit 1
fi
if [[ ! -x GR00T-WholeBodyControl/gear_sonic_deploy/deploy.sh ]]; then
    echo "[deploy] WBC deploy.sh not found / not executable." >&2
    exit 1
fi
if [[ ! -f "$KAPPA_CKPT" ]]; then
    echo "[deploy] KAPPA_CKPT not found: $KAPPA_CKPT" >&2
    exit 1
fi

# ---------- reclaim ports and orphan processes from prior runs ----------
# Port-bound holders (serve/bridge/kbd) from a prior run can survive across
# launches and the next start dies on EADDRINUSE. Mirrors the cleanup step
# in scripts/eval/run_kappa_sim_test.sh, minus the sim-only entries.
#
# We deliberately do NOT touch the WBC C++ deploy binary here: on real
# hardware it may still be holding motor PD state, and an unattended pkill
# is more dangerous than letting the next launch fail loudly. If a stale
# C++ deploy is running, the operator should kill it explicitly.
#
# Skip the whole step with SKIP_PORT_RECLAIM=1 (e.g. if you intentionally
# started something on these ports outside this script and want it alive).
if [[ "${SKIP_PORT_RECLAIM:-0}" != "1" ]]; then
    PORTS_TO_RECLAIM=("$WS_PORT" "$ACTION_PORT" "$STATE_PORT" "$KEYBOARD_PORT")
    need_kill=0
    for port in "${PORTS_TO_RECLAIM[@]}"; do
        pids=$(fuser "$port/tcp" 2>/dev/null | tr -s ' ' || true)
        if [[ -n "$pids" ]]; then
            echo "[deploy] reclaiming port $port (pids:$pids)" >&2
            fuser -k -TERM "$port/tcp" >/dev/null 2>&1 || true
            need_kill=1
        fi
    done
    if (( need_kill )); then
        sleep 1
        for port in "${PORTS_TO_RECLAIM[@]}"; do
            fuser -k -KILL "$port/tcp" >/dev/null 2>&1 || true
        done
    fi

    # Detached python children — match by full cmdline. Patterns are tight
    # enough that collisions with unrelated jobs are unlikely.
    ORPHAN_PATTERNS=(
        "kappa_serve_rtc\.py"
        "run_kappa_inference\.py"
        "keyboard_pub\.py --port $KEYBOARD_PORT"
    )
    need_kill=0
    for pat in "${ORPHAN_PATTERNS[@]}"; do
        pids=$(pgrep -f "$pat" 2>/dev/null | tr '\n' ' ' || true)
        if [[ -n "$pids" ]]; then
            echo "[deploy] reclaiming orphans /$pat/ (pids:${pids% })" >&2
            pkill -TERM -f "$pat" 2>/dev/null || true
            need_kill=1
        fi
    done
    if (( need_kill )); then
        sleep 1
        for pat in "${ORPHAN_PATTERNS[@]}"; do
            pkill -KILL -f "$pat" 2>/dev/null || true
        done
    fi
fi

PY=.venv/bin/python
REPO=$(pwd)

SERVE_CMD="cd $REPO && $PY src/inference/kappa_serve_rtc.py \
  --weights '$KAPPA_CKPT' \
  --backbone '$KAPPA_BACKBONE' \
  ${KAPPA_PROPRIO_STATS:+--proprio-stats '$KAPPA_PROPRIO_STATS' }\
  --normalize-mode bounds_q99 \
  --device '$KAPPA_DEVICE' \
  --ctrl-hz $CTRL_HZ \
  --host $KAPPA_HOST --port $WS_PORT"

WBC_CMD="cd $REPO/GR00T-WholeBodyControl/gear_sonic_deploy && \
  ./deploy.sh --input-type $WBC_INPUT_TYPE --hand-type $HAND_TYPE $WBC_MODE"

BRIDGE_CMD="cd $REPO && $PY scripts/deploy/run_kappa_inference.py \
  --ws-host $WS_CLIENT_HOST --ws-port $WS_PORT \
  --camera-host $CAMERA_HOST --camera-port $CAMERA_PORT \
  --state-zmq-host $STATE_HOST --state-zmq-port $STATE_PORT \
  --action-zmq-host $ACTION_HOST --action-zmq-port $ACTION_PORT \
  --keyboard-zmq-host $KEYBOARD_HOST --keyboard-zmq-port $KEYBOARD_PORT \
  --prompt '$BRIDGE_PROMPT' \
  --hand-type $HAND_TYPE \
  --tick-hz $CTRL_HZ${LOG_DIR:+ --log-dir '$LOG_DIR'}"

KBD_CMD="cd $REPO && $PY scripts/deploy/keyboard_pub.py --port $KEYBOARD_PORT"

cat <<EOF
[deploy] pre-flight passed and ports reclaimed.
[deploy] Open FOUR terminals and run one command in each, in this order:

# ---- Terminal 1: serve (Kappa WebSocket inference server) ----
# Wait until it prints "Listening on $KAPPA_HOST:$WS_PORT" before starting the bridge.
$SERVE_CMD

# ---- Terminal 2: wbc (C++ deploy) ----
# Manually answer 'y' at the "Proceed with deployment? [Y/n]:" prompt.
# Wait for "Init Done", then allow ~5s for the torso-IMU (rt/secondary_imu)
# DDS handshake before pressing 'k' in Terminal 4.
$WBC_CMD

# ---- Terminal 3: bridge (Kappa <-> WBC bridge) ----
# Start AFTER Terminal 1 is listening on $WS_PORT.
$BRIDGE_CMD

# ---- Terminal 4: kbd (keyboard publisher) ----
# Drive the pipeline manually by typing single keys (no Enter needed):
#   k = start C++ control loop in PLANNER mode (MOTORS ENGAGE on real hw)
#   i = publish LATENT_INITIAL_MOTION_TOKEN + switch C++ loop to POSE mode
#   p = unpause the bridge tick loop
$KBD_CMD

SAFETY (real hardware): pressing 'k' engages G1's leg motors. Position the
robot safely, hold the e-stop, and only press 'k' AFTER WBC has printed
"Init Done" and ~5s have elapsed for the torso-IMU DDS handshake.
EOF
