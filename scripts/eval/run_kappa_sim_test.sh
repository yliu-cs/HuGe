#!/usr/bin/env bash
# Closed-loop simulation test for the Kappa stack.
#
# Workflow:
#   * Model input   — by default, prerecorded offline-cache image + proprio
#                     are published in the same ZMQ formats the deploy bridge
#                     consumes.
#   * Action output — rendered inside MuJoCo (G1) through the real bridge /
#                     WBC deploy / sim loop path.
#
# Spawns up to seven tmux panes under session `kappa_sim_test`:
#   1. serve    — Kappa WebSocket inference server (src.inference.kappa_serve_rtc)
#   2. sim      — MuJoCo G1 loop (gear_sonic/scripts/run_sim_loop.py, offscreen)
#   3. wbc      — C++ deploy in `sim` mode (gear_sonic_deploy/deploy.sh ... sim)
#   4. cache    — offline dataset image/proprio publisher (sim_validate.py)
#   5. bridge   — Kappa <-> WBC bridge (scripts/deploy/run_kappa_inference.py)
#   6. kbd      — Keyboard publisher  (scripts/deploy/keyboard_pub.py)
#   7. recorder — On-screen viewer mp4 recorder (scripts/eval/x11_window_recorder.py,
#                 grabs the MuJoCo X11 window via ffmpeg x11grab; the viewer's
#                 default tracking camera is a third-person follow-cam, so the
#                 saved mp4 IS the third-person view. Gated on the same start
#                 sentinel as the offline-cache publisher, so ffmpeg launches
#                 exactly when frames begin and is SIGTERM'd by the auto-driver
#                 on "episode finished". Only when RECORD_VIDEO=1; writes
#                 $LOGDIR/sim.mp4)
#
# Pane output is mirrored to .runs/eval/<ts>/<window>.log via `script -q`.
#
# Required:
#   KAPPA_CKPT — absolute/repo-relative path to the Fine-Tune model.safetensors
# All other knobs default to the values used in the deploy guide; override via
# the env vars listed below.

set -euo pipefail

cd "$(dirname "$0")/../.."   # cd to Kappa repo root

# ---------- env defaults (override before invocation) ----------
: "${KAPPA_CKPT:=.runs/kappa/170R+85E_real.kappa.b1024.gpus8.2605241244/ckpt_1300/model.safetensors}"
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

# When SIM_MUJOCO_GL=egl on hosts that have BOTH the NVIDIA and Mesa EGL ICDs
# installed (e.g. RTX 5090 boxes with libEGL_mesa.so.0 and libEGL_nvidia.so.0
# both available), libglvnd may route to Mesa, which doesn't know how to
# drive the NVIDIA GPU and bails with:
#   libEGL warning: pci id for fd N: 10de:XXXX, driver (null)
#   libEGL warning: egl: failed to create dri2 screen
#   RuntimeError: Failed to make the EGL context current.
# Pinning __EGL_VENDOR_LIBRARY_FILENAMES to the NVIDIA ICD JSON forces glvnd
# to use NVIDIA's libEGL_nvidia.so.0 only. Set SIM_EGL_VENDOR= (empty) to
# disable the override; set to a different path to use Mesa or a custom ICD.
if [[ -z "${SIM_EGL_VENDOR+x}" ]]; then
    if [[ -f /usr/share/glvnd/egl_vendor.d/10_nvidia.json ]]; then
        SIM_EGL_VENDOR=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
    else
        SIM_EGL_VENDOR=
    fi
fi

# Resolve SIM_PY to an absolute path so it survives `cd` into the submodule.
if [[ "$SIM_PY" != /* ]]; then
    SIM_PY="$(pwd)/$SIM_PY"
fi

# C++ deploy uses the same launcher surface as scripts/deploy/run_kappa_deploy.sh,
# with WBC_MODE forced to sim for this orchestrator.
: "${WBC_MODE:=sim}"
: "${WBC_INPUT_TYPE:=zmq_manager}"

# Offline-cache input source. This is the default for sim_validate.py because
# it tests Kappa with the same dataset image/proprio inputs as offline eval
# while preserving the deploy bridge/WBC/sim action path.
: "${OFFLINE_CACHE_INPUT:=1}"
: "${OFFLINE_DATASET:=proprietary}"
: "${OFFLINE_DATA_ROOT:=resource/dataset/proprietary/real/2026-05-23}"
: "${OFFLINE_PROPRIO_CACHE_ROOT:=}"
: "${OFFLINE_EPISODE_INDEX:=123}"
: "${OFFLINE_SEED:=0}"
: "${OFFLINE_CAMERA_PORT:=6555}"
: "${OFFLINE_STATE_PORT:=6557}"

if [[ "$OFFLINE_CACHE_INPUT" == "1" ]]; then
    DEFAULT_CAMERA_PORT="$OFFLINE_CAMERA_PORT"
    DEFAULT_STATE_PORT="$OFFLINE_STATE_PORT"
else
    DEFAULT_CAMERA_PORT=5555
    DEFAULT_STATE_PORT=5557
fi

# Bridge wiring. In offline-cache mode, the bridge reads camera/state from the
# cache publisher. Set OFFLINE_CACHE_INPUT=0 to read from external/sim sources.
: "${WS_CLIENT_HOST:=localhost}"
: "${CAMERA_HOST:=localhost}"
: "${CAMERA_PORT:=$DEFAULT_CAMERA_PORT}"
: "${STATE_HOST:=localhost}"
: "${STATE_PORT:=$DEFAULT_STATE_PORT}"
: "${ACTION_HOST:=0.0.0.0}"
: "${ACTION_PORT:=5556}"
: "${WBC_STATE_PORT:=5557}"
: "${KEYBOARD_HOST:=localhost}"
: "${KEYBOARD_PORT:=5580}"
: "${BRIDGE_PROMPT:=Grab the cola, turn left, throw into the trash bin}"

: "${SESSION:=kappa_sim_test}"

# Auto-cleanup once the offline-cache publisher reports "episode finished".
# Disable by exporting AUTO_EXIT_ON_EPISODE_END=0 to keep the tmux session
# alive for post-run inspection. EPISODE_TIMEOUT_S is the max wall-clock the
# auto-driver waits for the end-of-episode signal before giving up.
# POST_EPISODE_WAIT_S is a grace window AFTER the cache publisher exits: the
# bridge stops emitting new actions, but MuJoCo's PD controller is still
# driving the joints toward the last commanded targets, so the actual robot
# pose lags the cache's final frame by ~1-5s. Recorder and tmux teardown are
# delayed by this much so the saved mp4 captures the trailing settle motion.
: "${AUTO_EXIT_ON_EPISODE_END:=1}"
: "${EPISODE_TIMEOUT_S:=600}"
: "${POST_EPISODE_WAIT_S:=5}"

# Sim video recording. When RECORD_VIDEO=1 the MuJoCo on-screen viewer
# window is captured to $LOGDIR/sim.mp4 via ffmpeg x11grab. We DON'T use the
# EGL offscreen renderer for the recording because on NVIDIA GLX (viewer) +
# EGL (offscreen) coexisting in the same process fails eglMakeCurrent; this
# also keeps the live MuJoCo window visible. Set RECORD_VIDEO=0 to disable.
# VIDEO_WINDOW_NAME / VIDEO_WINDOW_CLASS are passed to x11_window_recorder.py
# for matching; the defaults match mujoco.viewer.launch_passive. The recorder
# locates the window early but does not launch ffmpeg until OFFLINE_START_FILE
# is touched (same sentinel that releases the offline-cache publisher), so
# the mp4 covers exactly the data-publishing window.
: "${RECORD_VIDEO:=1}"
: "${VIDEO_FPS:=$CTRL_HZ}"
: "${VIDEO_WINDOW_NAME:=MuJoCo}"
: "${VIDEO_WINDOW_CLASS:=MuJoCo}"
: "${VIDEO_DISPLAY:=${DISPLAY:-:0}}"

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
#   1. Port holders   — serve/bridge/kbd/sim/cache publishers bind their ports.
#   2. Detached kids  — MuJoCo's offscreen renderer (multiprocessing.spawn)
#                       calls os.setsid() and survives even though it does
#                       not hold any port. Same goes for GLFW/EGL helper
#                       processes spawned by the sim and viewer threads.
# Skip the whole step with SKIP_PORT_RECLAIM=1 (e.g. if you started something
# unrelated on these ports/scripts manually and want to keep it alive).
if [[ "${SKIP_PORT_RECLAIM:-0}" != "1" ]]; then
    # (1) Port-bound holders.
    PORTS_TO_RECLAIM=(
        "$WS_PORT" "$SIM_CAMERA_PORT" "$ACTION_PORT" "$STATE_PORT" "$WBC_STATE_PORT"
        "$KEYBOARD_PORT" "$OFFLINE_CAMERA_PORT" "$OFFLINE_STATE_PORT"
    )
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
        "x11_window_recorder\.py"
        "zmq_video_recorder\.py"
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

LOGDIR=.runs/eval/$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOGDIR"
echo "[sim-test] logs -> $LOGDIR"
LOGDIR_ABS="$(pwd)/$LOGDIR"

if [[ "$RECORD_VIDEO" == "1" ]]; then
    : "${VIDEO_PATH:=$LOGDIR_ABS/sim.mp4}"
else
    : "${VIDEO_PATH:=}"
fi

if [[ "$OFFLINE_CACHE_INPUT" == "1" ]]; then
    : "${OFFLINE_START_FILE:=$LOGDIR_ABS/offline_cache.start}"
    rm -f "$OFFLINE_START_FILE"
else
    : "${OFFLINE_START_FILE:=}"
fi

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
REPO=$(pwd)

# Server: same recipe as run_kappa_deploy.sh.
SERVE_CMD="cd $REPO && $PY src/inference/kappa_serve_rtc.py \
  --weights '$KAPPA_CKPT' \
  --backbone '$KAPPA_BACKBONE' \
  --proprio-stats '$KAPPA_PROPRIO_STATS' \
  --normalize-mode bounds_q99 \
  --device '$KAPPA_DEVICE' \
  --ctrl-hz $CTRL_HZ \
  --host $KAPPA_HOST --port $WS_PORT"

# Sim loop GL backend selection.
#
# The on-screen MuJoCo viewer is ALWAYS enabled (the user wants the live
# window). The offscreen EGL renderer + image publisher is enabled only
# when the bridge actually needs MuJoCo-rendered frames as model input,
# i.e. OFFLINE_CACHE_INPUT=0 AND CAMERA_HOST=localhost (full-sim mode).
# If CAMERA_HOST points elsewhere (e.g. a real robot NUC providing
# ego_view over the network), publishing sim frames is pointless and
# actively harmful — the current G1 scene has no `head_camera` MJCF
# node, so offscreen rendering aborts the sim on startup.
# We also deliberately do NOT enable it for video recording: on NVIDIA,
# GLX (viewer) and EGL (offscreen) coexisting in the same process fails
# eglMakeCurrent. Video recording is handled by
# scripts/eval/x11_window_recorder.py grabbing the on-screen window.
SIM_PUBLISH=0
if [[ "$OFFLINE_CACHE_INPUT" != "1" && "$CAMERA_HOST" == "localhost" ]]; then
  SIM_PUBLISH=1
  echo "[sim-test] WARNING: OFFLINE_CACHE_INPUT=0 + CAMERA_HOST=localhost -> bridge needs EGL sim frames; the on-screen viewer will fail on NVIDIA because GLX+EGL cannot share a process. Set OFFLINE_CACHE_INPUT=1 to record from the viewer instead." >&2
elif [[ "$OFFLINE_CACHE_INPUT" != "1" ]]; then
  echo "[sim-test] CAMERA_HOST=$CAMERA_HOST (non-local) -> sim image publishing disabled; bridge will subscribe to the remote camera." >&2
fi
SIM_EGL_ENV=""
if (( SIM_PUBLISH )) && [[ -n "$SIM_EGL_VENDOR" ]]; then
  SIM_EGL_ENV="__EGL_VENDOR_LIBRARY_FILENAMES=$SIM_EGL_VENDOR "
fi
SIM_CMD="cd $REPO/GR00T-WholeBodyControl && \
  ${SIM_EGL_ENV}MUJOCO_GL=$SIM_MUJOCO_GL PYOPENGL_PLATFORM=$SIM_MUJOCO_GL \
  $SIM_PY gear_sonic/scripts/run_sim_loop.py"
if (( SIM_PUBLISH )); then
  SIM_CMD="$SIM_CMD \
    --enable-image-publish \
    --enable-offscreen \
    --no-enable-onscreen \
    --camera-port $SIM_CAMERA_PORT"
fi

# C++ deploy: defaults to `sim` here. Matches the deploy guide's Terminal 3.
# `echo y |` auto-confirms the interactive "Proceed with deployment? [Y/n]:"
# prompt at deploy.sh:547 so the orchestrator can run unattended.
WBC_CMD="cd $REPO/GR00T-WholeBodyControl/gear_sonic_deploy && \
  echo y | ./deploy.sh --input-type $WBC_INPUT_TYPE $WBC_MODE"

# Bridge: identical to the real-robot bridge, but defaults pull camera from
# the real G1 (CAMERA_HOST=192.168.123.164). Override CAMERA_HOST=localhost
# to drive the bridge entirely from the MuJoCo-rendered camera instead.
BRIDGE_CMD="cd $REPO && $PY scripts/deploy/run_kappa_inference.py \
  --ws-host $WS_CLIENT_HOST --ws-port $WS_PORT \
  --camera-host $CAMERA_HOST --camera-port $CAMERA_PORT \
  --state-zmq-host $STATE_HOST --state-zmq-port $STATE_PORT \
  --action-zmq-host $ACTION_HOST --action-zmq-port $ACTION_PORT \
  --keyboard-zmq-host $KEYBOARD_HOST --keyboard-zmq-port $KEYBOARD_PORT \
  --prompt '$BRIDGE_PROMPT' \
  --tick-hz $CTRL_HZ"

KBD_CMD="cd $REPO && $PY scripts/deploy/keyboard_pub.py --port $KEYBOARD_PORT"

CACHE_CMD="cd $REPO && $PY scripts/eval/sim_validate.py \
  --offline-cache-publisher \
  --dataset '$OFFLINE_DATASET' \
  --data-root '$OFFLINE_DATA_ROOT' \
  --episode-index $OFFLINE_EPISODE_INDEX \
  --seed $OFFLINE_SEED \
  --camera-port $OFFLINE_CAMERA_PORT \
  --state-port $OFFLINE_STATE_PORT \
  --ctrl-hz $CTRL_HZ"
if [[ -n "$OFFLINE_PROPRIO_CACHE_ROOT" ]]; then
  CACHE_CMD="$CACHE_CMD --proprio-cache-root '$OFFLINE_PROPRIO_CACHE_ROOT'"
fi
if [[ -n "$OFFLINE_START_FILE" ]]; then
  CACHE_CMD="$CACHE_CMD --start-file '$OFFLINE_START_FILE'"
fi

RECORDER_CMD="cd $REPO && DISPLAY=$VIDEO_DISPLAY $PY scripts/eval/x11_window_recorder.py \
  --output '$VIDEO_PATH' \
  --fps $VIDEO_FPS \
  --display '$VIDEO_DISPLAY' \
  --window-name '$VIDEO_WINDOW_NAME' \
  --window-class '$VIDEO_WINDOW_CLASS'"
# Same start sentinel as the offline-cache publisher so the recorder begins
# the moment data publishing begins (skipping the WBC warmup / k-i-p prelude).
if [[ -n "$OFFLINE_START_FILE" ]]; then
  RECORDER_CMD="$RECORDER_CMD --start-file '$OFFLINE_START_FILE'"
fi

# ---------- spawn ----------
# Single window, tiled panes so all logs are visible simultaneously.
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
if [[ "$OFFLINE_CACHE_INPUT" == "1" ]]; then
    tmux split-window -t "$SESSION:$WINDOW" \
        "script -q -c \"$CACHE_CMD\" $LOGDIR/cache.log"
    tmux select-layout -t "$SESSION:$WINDOW" tiled
fi
tmux split-window -t "$SESSION:$WINDOW" \
    "script -q -c \"$BRIDGE_CMD\" $LOGDIR/bridge.log"
tmux select-layout -t "$SESSION:$WINDOW" tiled
tmux split-window -t "$SESSION:$WINDOW" \
    "script -q -c \"$KBD_CMD\" $LOGDIR/kbd.log"
tmux select-layout -t "$SESSION:$WINDOW" tiled
# Recorder appended last so kbd / bridge / cache pane indices stay stable
# and the auto-driver's targeting math doesn't shift.
if [[ "$RECORD_VIDEO" == "1" ]]; then
    tmux split-window -t "$SESSION:$WINDOW" \
        "script -q -c \"$RECORDER_CMD\" $LOGDIR/recorder.log"
    tmux select-layout -t "$SESSION:$WINDOW" tiled
fi
tmux set-option -t "$SESSION" pane-border-status top
tmux select-pane -t "$SESSION:$WINDOW.0" -T "serve"
tmux select-pane -t "$SESSION:$WINDOW.1" -T "sim"
tmux select-pane -t "$SESSION:$WINDOW.2" -T "wbc"
if [[ "$OFFLINE_CACHE_INPUT" == "1" ]]; then
    tmux select-pane -t "$SESSION:$WINDOW.3" -T "cache"
    tmux select-pane -t "$SESSION:$WINDOW.4" -T "bridge"
    tmux select-pane -t "$SESSION:$WINDOW.5" -T "kbd"
    CACHE_PANE=3
    BRIDGE_PANE=4
    KBD_PANE=5
    WBC_PANE=2
    RECORDER_PANE=$([[ "$RECORD_VIDEO" == "1" ]] && echo 6 || echo "")
else
    tmux select-pane -t "$SESSION:$WINDOW.3" -T "bridge"
    tmux select-pane -t "$SESSION:$WINDOW.4" -T "kbd"
    CACHE_PANE=
    BRIDGE_PANE=3
    KBD_PANE=4
    WBC_PANE=2
    RECORDER_PANE=$([[ "$RECORD_VIDEO" == "1" ]] && echo 5 || echo "")
fi
if [[ -n "$RECORDER_PANE" ]]; then
    tmux select-pane -t "$SESSION:$WINDOW.$RECORDER_PANE" -T "recorder"
fi
# Focus the kbd pane so interactive keypresses land there by default.
tmux select-pane -t "$SESSION:$WINDOW.$KBD_PANE"

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
AUTO_SCRIPT="$LOGDIR/auto_driver.sh"
cat > "$AUTO_SCRIPT" <<EOF
#!/usr/bin/env bash
set -u

SESSION="$SESSION"
WINDOW="$WINDOW"
WBC_PANE="$WBC_PANE"
BRIDGE_PANE="$BRIDGE_PANE"
KBD_PANE="$KBD_PANE"
CACHE_PANE="$CACHE_PANE"
OFFLINE_CACHE_INPUT="$OFFLINE_CACHE_INPUT"
OFFLINE_START_FILE="$OFFLINE_START_FILE"
EB_FILE="$EB_FILE"
EB_FILE_BAK="$EB_FILE_BAK"
NEEDS_RESTORE="$NEEDS_RESTORE"
AUTO_EXIT_ON_EPISODE_END="$AUTO_EXIT_ON_EPISODE_END"
EPISODE_TIMEOUT_S="$EPISODE_TIMEOUT_S"
POST_EPISODE_WAIT_S="$POST_EPISODE_WAIT_S"
RECORD_VIDEO="$RECORD_VIDEO"
VIDEO_PATH="$VIDEO_PATH"
KEYBOARD_PORT="$KEYBOARD_PORT"
WS_PORT="$WS_PORT"
SIM_CAMERA_PORT="$SIM_CAMERA_PORT"
ACTION_PORT="$ACTION_PORT"
STATE_PORT="$STATE_PORT"
WBC_STATE_PORT="$WBC_STATE_PORT"
OFFLINE_CAMERA_PORT="$OFFLINE_CAMERA_PORT"
OFFLINE_STATE_PORT="$OFFLINE_STATE_PORT"

{
    echo "[auto] started at \$(date -Is)"
    sleep 15
    if (( NEEDS_RESTORE )) && [[ -f "\$EB_FILE_BAK" ]]; then
        mv "\$EB_FILE_BAK" "\$EB_FILE" \
            && echo "[auto] restored \$EB_FILE" \
            || echo "[auto] failed to restore \$EB_FILE"
    fi

    deadline=\$(( \$(date +%s) + 180 ))
    while true; do
        if tmux capture-pane -t "\$SESSION:\$WINDOW.\$WBC_PANE" -p -J -S - 2>/dev/null \
            | grep -q "Init Done"; then
            break
        fi
        if (( \$(date +%s) >= deadline )); then
            echo "[auto] timed out waiting for 'Init Done' in wbc pane; skipping auto-keys"
            exit 0
        fi
        # Bail out cleanly if the user killed the tmux session manually.
        if ! tmux has-session -t "\$SESSION" 2>/dev/null; then
            echo "[auto] tmux session '\$SESSION' gone; aborting auto-keys"
            exit 0
        fi
        sleep 0.5
    done
    echo "[auto] WBC reports Init Done; waiting for bridge WebSocket connection"

    deadline=\$(( \$(date +%s) + 180 ))
    while true; do
        if tmux capture-pane -t "\$SESSION:\$WINDOW.\$BRIDGE_PANE" -p -J -S - 2>/dev/null \
            | grep -q "\[bridge\] connected to"; then
            break
        fi
        if (( \$(date +%s) >= deadline )); then
            echo "[auto] timed out waiting for bridge WebSocket connection; skipping auto-keys"
            exit 0
        fi
        if ! tmux has-session -t "\$SESSION" 2>/dev/null; then
            echo "[auto] tmux session '\$SESSION' gone; aborting auto-keys"
            exit 0
        fi
        sleep 0.5
    done
    echo "[auto] bridge is connected; sending k -> i -> p to kbd pane"

    tmux send-keys -t "\$SESSION:\$WINDOW.\$KBD_PANE" 'k' Enter
    sleep 2
    tmux send-keys -t "\$SESSION:\$WINDOW.\$KBD_PANE" 'i' Enter
    sleep 2
    tmux send-keys -t "\$SESSION:\$WINDOW.\$KBD_PANE" 'p' Enter

    deadline=\$(( \$(date +%s) + 15 ))
    while true; do
        if tmux capture-pane -t "\$SESSION:\$WINDOW.\$BRIDGE_PANE" -p -J -S - 2>/dev/null \
            | grep -q "Resumed policy loop"; then
            break
        fi
        if (( \$(date +%s) >= deadline )); then
            echo "[auto] timed out waiting for bridge resume; offline cache remains blocked"
            exit 0
        fi
        if ! tmux has-session -t "\$SESSION" 2>/dev/null; then
            echo "[auto] tmux session '\$SESSION' gone; aborting offline cache release"
            exit 0
        fi
        sleep 0.1
    done
    if [[ -n "\$OFFLINE_START_FILE" ]]; then
        mkdir -p "\$(dirname "\$OFFLINE_START_FILE")"
        touch "\$OFFLINE_START_FILE"
        echo "[auto] released offline cache publisher: \$OFFLINE_START_FILE"
    fi
    echo "[auto] auto-sequence complete; pipeline should now be running"

    # ---- shutdown on episode end ------------------------------------------
    # The offline-cache publisher prints "episode finished" once at
    # sim_validate.py:130 and then exits. Use it as the cue to tear down the
    # whole tmux session + any detached MuJoCo/EGL/multiprocessing children
    # (same orphan classes the pre-flight block reclaims).
    if [[ "\$OFFLINE_CACHE_INPUT" != "1" || "\$AUTO_EXIT_ON_EPISODE_END" != "1" || -z "\$CACHE_PANE" ]]; then
        echo "[auto] auto-exit disabled (OFFLINE_CACHE_INPUT=\$OFFLINE_CACHE_INPUT AUTO_EXIT_ON_EPISODE_END=\$AUTO_EXIT_ON_EPISODE_END); leaving session up"
        exit 0
    fi

    deadline=\$(( \$(date +%s) + EPISODE_TIMEOUT_S ))
    while true; do
        if tmux capture-pane -t "\$SESSION:\$WINDOW.\$CACHE_PANE" -p -J -S - 2>/dev/null \
            | grep -q "episode finished"; then
            echo "[auto] cache reports episode finished; tearing down"
            break
        fi
        if (( \$(date +%s) >= deadline )); then
            echo "[auto] timed out waiting for episode-finished after \${EPISODE_TIMEOUT_S}s; tearing down anyway"
            break
        fi
        if ! tmux has-session -t "\$SESSION" 2>/dev/null; then
            echo "[auto] tmux session '\$SESSION' gone before episode end; nothing to clean"
            exit 0
        fi
        sleep 0.5
    done

    # Pause the bridge as soon as the episode ends so it stops emitting
    # actions sourced from stale (or no longer published) camera frames.
    # 'p' toggles run/pause in the bridge keyboard handler; the initial
    # k -> i -> p sequence above leaves the bridge running, so this second
    # 'p' flips it back to paused. With Plan A's 50Hz producer/consumer
    # bridge, every second of "stale grace" would otherwise pump ~50 weird
    # actions into WBC.
    tmux send-keys -t "\$SESSION:\$WINDOW.\$KBD_PANE" 'p' Enter 2>/dev/null \
        && echo "[auto] paused bridge (sent 'p' to kbd pane)" \
        || echo "[auto] could not send pause to kbd pane"

    # Grace window: cache publisher has exited, but the last 1-5s of motion
    # is still being executed by MuJoCo PD + WBC. Keep the recorder running
    # and the bridge/WBC/sim panes alive so the saved mp4 + logs capture the
    # settle motion. Skip by exporting POST_EPISODE_WAIT_S=0.
    if (( POST_EPISODE_WAIT_S > 0 )); then
        echo "[auto] sleeping \${POST_EPISODE_WAIT_S}s post-episode for PD/sim to settle"
        for _ in \$(seq 1 \$POST_EPISODE_WAIT_S); do
            if ! tmux has-session -t "\$SESSION" 2>/dev/null; then
                echo "[auto] tmux session '\$SESSION' gone during post-episode wait; skipping rest"
                break
            fi
            sleep 1
        done
    fi

    # Finalize the recorder FIRST. x11_window_recorder.py traps SIGTERM and
    # forwards SIGINT to ffmpeg, which writes the mp4 trailer (moov atom)
    # before exiting. If we kill the tmux session first, the recorder dies
    # via SIGHUP through 'script -q' without running its handler and the
    # mp4 ends up unplayable.
    if [[ "\$RECORD_VIDEO" == "1" ]]; then
        pkill -TERM -f "x11_window_recorder\\.py" 2>/dev/null && \
            echo "[auto] sent SIGTERM to recorder; waiting for mp4 finalize" || \
            echo "[auto] no recorder process found to SIGTERM"
        # Poll up to 10s for the recorder + ffmpeg to exit (ffmpeg's trailer
        # write typically returns in 1-2s; budget extra for slower disks
        # and the ffmpeg signal-handling delay).
        for _ in 1 2 3 4 5 6 7 8 9 10; do
            if ! pgrep -f "x11_window_recorder\\.py" >/dev/null 2>&1; then
                break
            fi
            sleep 1
        done
        if [[ -n "\$VIDEO_PATH" && -s "\$VIDEO_PATH" ]]; then
            echo "[auto] video saved: \$VIDEO_PATH (\$(stat -c%s "\$VIDEO_PATH") bytes)"
        elif [[ -n "\$VIDEO_PATH" ]]; then
            echo "[auto] WARNING: \$VIDEO_PATH missing or empty"
        fi
    fi

    # Give panes ~2s to flush their last log lines before we yank the session.
    sleep 2
    tmux kill-session -t "\$SESSION" 2>/dev/null && echo "[auto] killed tmux session '\$SESSION'" \
        || echo "[auto] tmux kill-session reported no session (already gone)"

    # Mirror pre-flight cleanup: ports first, then orphan process trees that
    # survive SIGHUP through 'script -q' (MuJoCo offscreen renderer + GLFW/EGL
    # helpers set their own session via os.setsid).
    PORTS_TO_RECLAIM=(
        "\$WS_PORT" "\$SIM_CAMERA_PORT" "\$ACTION_PORT" "\$STATE_PORT" "\$WBC_STATE_PORT"
        "\$KEYBOARD_PORT" "\$OFFLINE_CAMERA_PORT" "\$OFFLINE_STATE_PORT"
    )
    for port in "\${PORTS_TO_RECLAIM[@]}"; do
        fuser -k -TERM "\$port/tcp" >/dev/null 2>&1 || true
    done
    ORPHAN_PATTERNS=(
        "kappa_serve_rtc\\.py"
        "run_kappa_inference\\.py"
        "keyboard_pub\\.py --port \$KEYBOARD_PORT"
        "run_sim_loop\\.py"
        "sim_validate\\.py"
        "x11_window_recorder\\.py"
        "zmq_video_recorder\\.py"
        "\\.venv_sim/bin/python .*multiprocessing"
    )
    for pat in "\${ORPHAN_PATTERNS[@]}"; do
        pkill -TERM -f "\$pat" 2>/dev/null || true
    done
    sleep 1
    for port in "\${PORTS_TO_RECLAIM[@]}"; do
        fuser -k -KILL "\$port/tcp" >/dev/null 2>&1 || true
    done
    for pat in "\${ORPHAN_PATTERNS[@]}"; do
        pkill -KILL -f "\$pat" 2>/dev/null || true
    done
    echo "[auto] cleanup complete"
} >> "$AUTO_LOG" 2>&1
EOF
chmod +x "$AUTO_SCRIPT"
tmux run-shell -b "$AUTO_SCRIPT"
echo "[sim-test] auto-driver running in background; tail $AUTO_LOG for progress"

cat <<EOF
[sim-test] session '$SESSION' launched (tiled panes in window '$WINDOW';
           input camera=$CAMERA_HOST:$CAMERA_PORT, state=$STATE_HOST:$STATE_PORT,
           sim render port=$SIM_CAMERA_PORT, offline_cache=$OFFLINE_CACHE_INPUT).
           offline cache starts only after WBC Init Done + k/i/p
           auto-driver: yaml elastic-band restored + k/i/p sent on 'Init Done',
                        and on 'episode finished' the whole session + MuJoCo
                        viewer/children are torn down after a
                        POST_EPISODE_WAIT_S=${POST_EPISODE_WAIT_S}s grace window so
                        MuJoCo PD can settle on the final pose
                        (AUTO_EXIT_ON_EPISODE_END=$AUTO_EXIT_ON_EPISODE_END,
                        timeout=${EPISODE_TIMEOUT_S}s; see $LOGDIR/auto.log)
           attach:        tmux attach -t $SESSION
           switch pane:   Ctrl-b o   (cycle)   Ctrl-b <arrow>   (directional)
           zoom pane:     Ctrl-b z   (toggle fullscreen for the focused pane)
           resize pane:   Ctrl-b Alt-<arrow>
           detach:        Ctrl-b d
           stop all:      tmux kill-session -t $SESSION
           logs:          $LOGDIR/{serve,sim,wbc,cache,bridge,kbd,recorder,auto}.log
           video:         $([[ "$RECORD_VIDEO" == "1" ]] && echo "$VIDEO_PATH (RECORD_VIDEO=1, x11grab name~='$VIDEO_WINDOW_NAME' class='$VIDEO_WINDOW_CLASS' on $VIDEO_DISPLAY @ ${VIDEO_FPS}fps)" || echo "disabled (RECORD_VIDEO=0)")
EOF
