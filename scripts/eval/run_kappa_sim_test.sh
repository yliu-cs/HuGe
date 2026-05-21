#!/usr/bin/env bash
# Closed-loop simulation test for the Kappa stack.
#
# Workflow (mirrors the "仿真测试" section of `G1 真机部署指南.pdf`):
#   * Camera input  — pulled from the REAL robot's ZMQ camera server
#                     (override CAMERA_HOST/CAMERA_PORT to point elsewhere).
#   * Action output — rendered inside MuJoCo (G1) instead of the real robot.
#
# Spawns five tmux windows under session `kappa_sim_test`:
#   1. serve  — Kappa WebSocket inference server (src.inference.kappa_serve_rtc)
#   2. sim    — MuJoCo G1 loop (gear_sonic/scripts/run_sim_loop.py, offscreen)
#   3. wbc    — C++ deploy in `sim` mode (gear_sonic_deploy/deploy.sh ... sim)
#   4. bridge — Kappa <-> WBC bridge (scripts/deploy/run_kappa_inference.py)
#   5. kbd    — Keyboard publisher  (scripts/deploy/keyboard_pub.py)
#
# Pane output is mirrored to .runs/sim_test/<ts>/<window>.log via `script -q`.
#
# Required:
#   KAPPA_CKPT — absolute/repo-relative path to the Fine-Tune model.safetensors
# All other knobs default to the values used in the deploy guide; override via
# the env vars listed below.

set -euo pipefail

cd "$(dirname "$0")/../.."   # cd to Kappa repo root

# ---------- env defaults (override before invocation) ----------
: "${KAPPA_CKPT:=.runs/kappa/exp.kappa.b1024.gpus8.2605181229/checkpoints/ckpt_5000/model.safetensors}"
: "${KAPPA_BACKBONE:=resource/model/Qwen3-VL-2B-Instruct}"
: "${KAPPA_PROPRIO_STATS:=resource/dataset/G1_WBT-cache/meta/g1_proprio_stats.json}"
: "${KAPPA_HOST:=0.0.0.0}"
: "${WS_PORT:=8765}"
: "${CTRL_HZ:=50}"
: "${KAPPA_DEVICE:=cuda:0}"

# MuJoCo sim loop — published image port is independent of the bridge's
# camera source. Default matches `run_sim_loop.py`.
: "${SIM_PY:=GR00T-WholeBodyControl/.venv_sim/bin/python}"
: "${SIM_CAMERA_PORT:=5555}"
: "${SIM_MUJOCO_GL:=egl}"

# Resolve SIM_PY to an absolute path so it survives `cd` into the submodule.
if [[ "$SIM_PY" != /* ]]; then
    SIM_PY="$(pwd)/$SIM_PY"
fi

# C++ deploy is always launched in `sim` mode for this orchestrator.
: "${WBC_INPUT_TYPE:=zmq_manager}"
: "${WBC_ZMQ_HOST:=localhost}"

# Bridge wiring. Defaults assume the user has a real G1 publishing camera
# frames at 192.168.123.164; override CAMERA_HOST=localhost to drive the
# bridge from the sim's own rendered images instead.
: "${WS_CLIENT_HOST:=localhost}"
: "${CAMERA_HOST:=192.168.123.164}"
: "${CAMERA_PORT:=5555}"
: "${STATE_HOST:=localhost}"
: "${STATE_PORT:=5557}"
: "${ACTION_HOST:=0.0.0.0}"
: "${ACTION_PORT:=5556}"
: "${KEYBOARD_HOST:=localhost}"
: "${KEYBOARD_PORT:=5580}"
: "${BRIDGE_PROMPT:=Walk to the table and push the scotch tape to the blue tape}"

: "${SESSION:=kappa_sim_test}"

# ---------- session guard ----------
if tmux has-session -t "$SESSION" 2>/dev/null; then
    cat >&2 <<EOF
[sim-test] tmux session '$SESSION' already exists.
           attach:  tmux attach -t $SESSION
           restart: tmux kill-session -t $SESSION && $0
EOF
    exit 1
fi

# ---------- pre-flight ----------
if [[ ! -x .venv/bin/python ]]; then
    echo "[sim-test] .venv/bin/python not found. Run \`uv sync\` first." >&2
    exit 1
fi
if [[ ! -x "$SIM_PY" ]]; then
    cat >&2 <<EOF
[sim-test] sim interpreter not found at: $SIM_PY
           Set up the MuJoCo venv from inside GR00T-WholeBodyControl:
               cd GR00T-WholeBodyControl
               bash install_scripts/install_mujoco_sim.sh
           Or override SIM_PY=... to point at an existing venv with
           gear_sonic[sim] installed.
EOF
    exit 1
fi
if [[ ! -x GR00T-WholeBodyControl/gear_sonic_deploy/deploy.sh ]]; then
    echo "[sim-test] WBC deploy.sh not found / not executable." >&2
    exit 1
fi
if [[ ! -f "$KAPPA_CKPT" ]]; then
    echo "[sim-test] KAPPA_CKPT not found: $KAPPA_CKPT" >&2
    exit 1
fi

# ---------- reclaim ports and orphan processes from prior runs ----------
# `tmux kill-session` does not reliably propagate SIGHUP through `script -q`
# to the inner Python. Two classes of orphans need to be cleaned:
#   1. Port holders   — serve/bridge/kbd/sim publisher bind 8765/5556/5580/5555.
#   2. Detached kids  — MuJoCo's offscreen renderer (multiprocessing.spawn)
#                       calls os.setsid() and survives even though it does
#                       not hold any port. Same goes for GLFW/EGL helper
#                       processes spawned by the sim and viewer threads.
# Skip the whole step with SKIP_PORT_RECLAIM=1 (e.g. if you started something
# unrelated on these ports/scripts manually and want to keep it alive).
if [[ "${SKIP_PORT_RECLAIM:-0}" != "1" ]]; then
    # (1) Port-bound holders.
    PORTS_TO_RECLAIM=("$WS_PORT" "$SIM_CAMERA_PORT" "$ACTION_PORT" "$STATE_PORT" "$KEYBOARD_PORT")
    need_kill=0
    for port in "${PORTS_TO_RECLAIM[@]}"; do
        pids=$(fuser "$port/tcp" 2>/dev/null | tr -s ' ' || true)
        if [[ -n "$pids" ]]; then
            echo "[sim-test] reclaiming port $port (pids:$pids)" >&2
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

    # (2) Detached process tree — match by full cmdline. Patterns are tight
    # enough that collisions with unrelated python jobs are unlikely.
    ORPHAN_PATTERNS=(
        "kappa_serve_rtc\.py"
        "run_kappa_inference\.py"
        "keyboard_pub\.py --port $KEYBOARD_PORT"
        "run_sim_loop\.py"
        "\.venv_sim/bin/python .*multiprocessing"
    )
    need_kill=0
    for pat in "${ORPHAN_PATTERNS[@]}"; do
        pids=$(pgrep -f "$pat" 2>/dev/null | tr '\n' ' ' || true)
        if [[ -n "$pids" ]]; then
            echo "[sim-test] reclaiming orphans /$pat/ (pids:${pids% })" >&2
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

LOGDIR=.runs/sim_test/$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOGDIR"
echo "[sim-test] logs -> $LOGDIR"

# ---------- elastic-band override ----------
# Goal: start G1 on the floor instead of suspended by the virtual spring at
# z=1. The deploy guide tells the user to press `9` in the MuJoCo viewer to
# toggle the band off (see ElasticBand.MujuocoKeyCallback). xdotool isn't
# installed and the viewer is owned by a `multiprocessing.spawn` child, so
# we can't reliably synthesize a GLFW keystroke from the orchestrator.
#
# We can't just set ENABLE_ELASTIC_BAND: False in the yaml either —
# base_sim.py:394 reads `self.elastic_band` unconditionally but only
# initializes it in the True branch, so flipping the yaml crashes the sim
# with `'DefaultEnv' object has no attribute 'elastic_band'`.
#
# Workaround: flip ElasticBand.__init__'s default `self.enable = True` to
# `False`. The band object is still constructed (no AttributeError), but
# sim_step() takes the zero-force branch on every tick — identical to what
# pressing `9` in the viewer does. The background watcher restores the
# original line once the sim has loaded (the class is read once at import).
EB_FILE="GR00T-WholeBodyControl/gear_sonic/utils/mujoco_sim/unitree_sdk2py_bridge.py"
EB_FILE_BAK="$EB_FILE.simtest_bak"
NEEDS_RESTORE=0
if [[ -f "$EB_FILE" ]] && grep -q '^        self.enable = True$' "$EB_FILE"; then
    cp "$EB_FILE" "$EB_FILE_BAK"
    sed -i 's/^        self.enable = True$/        self.enable = False  # auto-patched by run_kappa_sim_test.sh (restored after sim load)/' "$EB_FILE"
    NEEDS_RESTORE=1
    echo "[sim-test] patched ElasticBand.enable=False in $EB_FILE (will restore once sim has loaded)"
fi

PY=.venv/bin/python

# Server: same recipe as run_kappa_deploy.sh.
SERVE_CMD="$PY src/inference/kappa_serve_rtc.py \
  --weights '$KAPPA_CKPT' \
  --backbone '$KAPPA_BACKBONE' \
  --proprio-stats '$KAPPA_PROPRIO_STATS' \
  --normalize-mode bounds_q99 \
  --device '$KAPPA_DEVICE' \
  --ctrl-hz $CTRL_HZ \
  --host $KAPPA_HOST --port $WS_PORT"

# Sim loop: render offscreen and publish the rendered ego-view on
# tcp://0.0.0.0:$SIM_CAMERA_PORT for optional inspection.
# Run from GR00T-WholeBodyControl/ so the script's WBC yaml lookup works.
SIM_CMD="cd GR00T-WholeBodyControl && \
  MUJOCO_GL=$SIM_MUJOCO_GL PYOPENGL_PLATFORM=$SIM_MUJOCO_GL \
  $SIM_PY gear_sonic/scripts/run_sim_loop.py \
    --enable-image-publish \
    --enable-offscreen \
    --camera-port $SIM_CAMERA_PORT"

# C++ deploy: always `sim` here. Matches the deploy guide's Terminal 3.
# `echo y |` auto-confirms the interactive "Proceed with deployment? [Y/n]:"
# prompt at deploy.sh:547 so the orchestrator can run unattended.
WBC_CMD="cd GR00T-WholeBodyControl/gear_sonic_deploy && \
  echo y | ./deploy.sh --input-type $WBC_INPUT_TYPE --zmq-host $WBC_ZMQ_HOST sim"

# Bridge: identical to the real-robot bridge, but defaults pull camera from
# the real G1 (CAMERA_HOST=192.168.123.164). Override CAMERA_HOST=localhost
# to drive the bridge entirely from the MuJoCo-rendered camera instead.
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
# Single window, 5 tiled panes so all logs are visible simultaneously.
# Server pane first (model load + cuDNN warmup dominates startup); sim
# pane before WBC so the C++ deploy has a peer to handshake with over DDS.
WINDOW=main
tmux new-session -d -s "$SESSION" -n "$WINDOW" \
    "script -q -c \"$SERVE_CMD\" $LOGDIR/serve.log"
# Keep dead panes visible so the user can see why a pane exited instead of
# tmux silently removing it (which would leave fewer panes than expected).
tmux set-option -t "$SESSION" remain-on-exit on

# Tiny pause so subsequent pane logs don't race the server's bind log.
sleep 0.3

# Re-tile after EVERY split. Without this, default vertical splits keep
# halving the active pane height; on a standard 24-row terminal the
# 5th split would land in a 1-row pane and tmux silently rejects it.
tmux split-window -t "$SESSION:$WINDOW" \
    "script -q -c \"$SIM_CMD\" $LOGDIR/sim.log"
tmux select-layout -t "$SESSION:$WINDOW" tiled
tmux split-window -t "$SESSION:$WINDOW" \
    "script -q -c \"$WBC_CMD\" $LOGDIR/wbc.log"
tmux select-layout -t "$SESSION:$WINDOW" tiled
tmux split-window -t "$SESSION:$WINDOW" \
    "script -q -c \"$BRIDGE_CMD\" $LOGDIR/bridge.log"
tmux select-layout -t "$SESSION:$WINDOW" tiled
tmux split-window -t "$SESSION:$WINDOW" \
    "script -q -c \"$KBD_CMD\" $LOGDIR/kbd.log"
tmux select-layout -t "$SESSION:$WINDOW" tiled
tmux set-option -t "$SESSION" pane-border-status top
tmux select-pane -t "$SESSION:$WINDOW.0" -T "serve"
tmux select-pane -t "$SESSION:$WINDOW.1" -T "sim"
tmux select-pane -t "$SESSION:$WINDOW.2" -T "wbc"
tmux select-pane -t "$SESSION:$WINDOW.3" -T "bridge"
tmux select-pane -t "$SESSION:$WINDOW.4" -T "kbd"
# Focus the kbd pane so interactive keypresses land there by default.
tmux select-pane -t "$SESSION:$WINDOW.4"

# ---------- background automation ----------
# Runs detached from the orchestrator so the user gets their shell back.
#   1. Wait 15s, restore the yaml (sim has read it by then).
#   2. Poll wbc.log for "Init Done" (printed at g1_deploy_onnx_ref.cpp:2754)
#      with a 180s timeout.
#   3. Drive the bridge via the kbd pane:  k -> i -> p
#        k = start C++ control loop in PLANNER mode
#        i = publish LATENT_INITIAL_MOTION_TOKEN + switch C++ loop to POSE mode
#        p = unpause the bridge tick loop
#      Newlines from `Enter` are ignored by keyboard_pub.py (cbreak, single-
#      byte reads; SINGLE_KEYS excludes '\n').
AUTO_LOG="$LOGDIR/auto.log"
(
    sleep 15
    if (( NEEDS_RESTORE )) && [[ -f "$EB_FILE_BAK" ]]; then
        mv "$EB_FILE_BAK" "$EB_FILE" \
            && echo "[auto] restored $EB_FILE" \
            || echo "[auto] failed to restore $EB_FILE"
    fi

    # NOTE: `script -q` block-buffers its output file, so $WBC_LOG can lag
    # the live pane by hundreds of lines. We query the live pane via tmux
    # capture-pane (`-S -` = from the start of the scrollback) instead.
    deadline=$(( $(date +%s) + 180 ))
    while true; do
        if tmux capture-pane -t "$SESSION:$WINDOW.2" -p -S - 2>/dev/null \
            | grep -q "Init Done"; then
            break
        fi
        if (( $(date +%s) >= deadline )); then
            echo "[auto] timed out waiting for 'Init Done' in wbc pane; skipping auto-keys"
            exit 0
        fi
        # Bail out cleanly if the user killed the tmux session manually.
        if ! tmux has-session -t "$SESSION" 2>/dev/null; then
            echo "[auto] tmux session '$SESSION' gone; aborting auto-keys"
            exit 0
        fi
        sleep 0.5
    done
    echo "[auto] WBC reports Init Done; sending k -> i -> p to kbd pane"

    tmux send-keys -t "$SESSION:$WINDOW.4" 'k' Enter
    sleep 2
    tmux send-keys -t "$SESSION:$WINDOW.4" 'i' Enter
    sleep 2
    tmux send-keys -t "$SESSION:$WINDOW.4" 'p' Enter
    echo "[auto] auto-sequence complete; pipeline should now be running"
) >> "$AUTO_LOG" 2>&1 &
disown
echo "[sim-test] auto-driver running in background; tail $AUTO_LOG for progress"

cat <<EOF
[sim-test] session '$SESSION' launched (5 panes tiled in window '$WINDOW';
           camera=$CAMERA_HOST:$CAMERA_PORT, sim render port=$SIM_CAMERA_PORT).
           auto-driver: yaml elastic-band restored + k/i/p sent on 'Init Done'
                        (see $LOGDIR/auto.log)
           attach:        tmux attach -t $SESSION
           switch pane:   Ctrl-b o   (cycle)   Ctrl-b <arrow>   (directional)
           zoom pane:     Ctrl-b z   (toggle fullscreen for the focused pane)
           resize pane:   Ctrl-b Alt-<arrow>
           detach:        Ctrl-b d
           stop all:      tmux kill-session -t $SESSION
           logs:          $LOGDIR/{serve,sim,wbc,bridge,kbd,auto}.log
EOF
