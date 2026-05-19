# $\kappa_{0}$: A Foundation Vision-Language-Action Model for General Humanoid Robot Loco-Manipulation Control

claude --resume 31ae9900-a7da-415e-8a42-71657ececc36

## 🏠 Installation

```
git clone --recurse-submodules https://github.com/yliu-cs/Kappa.git
# (or, after a plain clone: `git submodule update --init --recursive`)
cd Kappa
uv sync
uv pip install flash_attn==2.7.4.post1 --no-build-isolation

mkdir -p resource/dataset
wget https://hf-mirror.com/hfd/hfd.sh && chmod a+x hfd.sh
./hfd.sh USC-PSI-Lab/humanoid-everyday --dataset
# Download EgoDex from [ml-egodex](https://github.com/apple/ml-egodex) to "resource/dataset/EgoDex"
# Download G1-WBT from [unifolm-wbt](https://huggingface.co/collections/unitreerobotics/unifolm-wbt-dataset) to "resource/dataset/G1_WBT"
mkdir -p resource/model
cd resource/model
wget https://hf-mirror.com/hfd/hfd.sh && chmod a+x hfd.sh
./hfd.sh Qwen/Qwen3-VL-2B-Instruct
mkdir -p sonic
./hfd.sh nvidia/GEAR-SONIC --local-dir ./sonic --include model_encoder.onnx
./hfd.sh nvidia/GEAR-SONIC --local-dir ./sonic --include model_decoder.onnx
./hfd.sh nvidia/GEAR-SONIC --local-dir ./sonic --include observation_config.yaml
```

## 🚀 Training

```
# Stage 1: Pre-Train on EgoDex
uv run python scripts/preprocess/egodex/build_index.py
tmux new -s pretrain_egodex -d "script -q -c './scripts/train/train_kappa.sh pretrain_egodex_kappa_config' pretrain_egodex.log"

# Stage 2: Post-Train on Humanoid-Everyday G1 + G1-WBT (SONIC tokens + RTC)
uv run python scripts/preprocess/humanoid-everyday/calc_hand_thresholds.py
tmux new -s calc_he_sonic -d "uv run python scripts/preprocess/humanoid-everyday/calc_sonic_tokens.py"
tmux new -s calc_wbt_sonic -d "uv run python scripts/preprocess/g1-wbt/calc_wbt_sonic_tokens.py"
tmux new -s posttrain_he_wbt -d "script -q -c './scripts/train/train_kappa.sh posttrain_he_wbt_kappa_config' posttrain_he_wbt.log"

# Stage 3: Fine-Tune on Proprietary Data (SONIC tokens)
tmux new -s calc_prop_sonic -d "uv run python scripts/preprocess/proprietary/calc_sonic_tokens.py"
tmux new -s finetune_prop -d "script -q -c './scripts/train/train_kappa.sh finetune_prop_ep_kappa_config' finetune_prop.log"

# Or run the full chain (pretrain -> posttrain -> finetune) end-to-end:
tmux new -s run_kappa -d "script -q -c './scripts/train/run_kappa.sh' run_kappa.log"
```

## 🎯 Evaluation

```
uv run python scripts/eval/sim_validate.py \
  --dataset proprietary \
  --checkpoint .runs/kappa/exp.kappa.b1024.gpus8.2605181229/checkpoints/ckpt_5000/model.safetensors \
  --heads sonic gt_sonic \
  --episode-indices 0 \
  --out-dir .runs/eval/$(date +%Y%m%d_%H%M%S)
```

## 🤖 Deployment

### Start the server

```
.venv/bin/python -m src.inference.kappa_serve_rtc \
  --weights .runs/kappa/<finetune-run>/checkpoints/ckpt_<step>/model.safetensors \
  --backbone resource/model/Qwen3-VL-2B-Instruct \
  --proprio-stats resource/dataset/G1_WBT-cache/meta/g1_proprio_stats.json \
  --normalize-mode bounds_q99 \
  --ctrl-hz 50 \
  --host 0.0.0.0 --port 8765
```

### Client → server payload (WebSocket text frame, JSON)

```jsonc
{
  "image":   "<base64-encoded PNG of the latest ego-view frame>",
  "text":    "perform the task",
  "proprio": [/* 64 RAW floats, see layout below */]
}
```

| Index    | Field      | Units    | Notes                                                                 |
| -------- | ---------- | -------- | --------------------------------------------------------------------- |
| `[0:29]` | `jp`       | rad      | Joint positions in **MuJoCo joint order** (see `MUJOCO_TO_ISAACLAB`). |
| `[29:31]`| `hand_bits`| binary   | 0/1 per gripper (left, right). Use `0,0` for fake-hand datasets.      |
| `[31:60]`| `jv`       | rad/s    | Finite-difference of `jp` at 50 Hz: `(jp[t] - jp[t-1]) * 50`.         |
| `[60:64]`| `quat_xyzw`| unit q   | Pelvis orientation, **xyzw** order.                                   |

### Server → client message (JSON)

```jsonc
{ "version": 1234, "sonic": [/* 66 floats */] }
```

`version` monotonically increases each control tick (1/50 s). The 66-D vector is `[sonic_latent(64) | hand_bits(2)]`.

### One-command launch (tmux orchestrator)

The four-terminal stack (Kappa server, WBC C++ deploy, bridge, keyboard) is bundled into a single tmux session by `scripts/deploy/run_kappa_deploy.sh`. The bridge (`scripts/deploy/run_kappa_inference.py`) plays the role of GR00T's `run_vla_inference.py`: it reads camera + state ZMQ, drives the Kappa WebSocket at 50 Hz, expands the server's 2 hand-bits into 7-DoF left/right hand joints via `G1GripperInverseKinematicsSolver`, and publishes Protocol-v4 `pose` messages on the action ZMQ socket consumed by `gear_sonic_deploy`.

```bash
KAPPA_CKPT=.runs/kappa/<finetune-run>/checkpoints/ckpt_<step>/model.safetensors \
./scripts/deploy/run_kappa_deploy.sh
tmux attach -t kappa_deploy
```

tmux windows (`Ctrl-b n` / `Ctrl-b p` to cycle, `Ctrl-b d` to detach):

1. `serve`  — `src.inference.kappa_serve_rtc` on port 8765
2. `wbc`    — `GR00T-WholeBodyControl/gear_sonic_deploy/deploy.sh --input-type zmq_manager real`
3. `bridge` — Kappa ↔ WBC bridge; auto-reconnects to the server with exponential backoff
4. `kbd`    — interactive keyboard publisher (ZMQ PUB on 5580)

Keyboard window controls:

| Key | Effect |
| --- | ------ |
| `p` | pause / resume bridge inference loop |
| `k` | start / stop C++ control loop (PLANNER mode) |
| `i` | send initial pose; switch to POSE mode; reset frame counter |
| `[` / `]` | toggle initial-pose left / right hand closed |
| `t` | enter a new language prompt (line entry; publishes `prompt:<text>`) |
| `c` / `s` / `f` | start / stop-success / stop-failure recording (data exporter) |
| `q` | quit the keyboard publisher only |

Stop everything: `tmux kill-session -t kappa_deploy`. Per-window logs land in `.runs/deploy/<timestamp>/{serve,wbc,bridge,kbd}.log`.

Env vars accepted by the orchestrator (defaults shown in the script):

- **Required**: `KAPPA_CKPT`
- **Model**: `KAPPA_BACKBONE`, `KAPPA_PROPRIO_STATS`, `KAPPA_DEVICE`, `WS_PORT`, `KAPPA_HOST`, `CTRL_HZ`
- **WBC**: `WBC_MODE` (`real` or `sim`), `WBC_INPUT_TYPE`
- **ZMQ wiring**: `CAMERA_HOST`/`CAMERA_PORT`, `STATE_HOST`/`STATE_PORT`, `ACTION_HOST`/`ACTION_PORT`, `KEYBOARD_HOST`/`KEYBOARD_PORT`
- **Bridge**: `WS_CLIENT_HOST`, `BRIDGE_PROMPT`

Smoke testing without hardware: set `WBC_MODE=sim` to swap the C++ deploy into MuJoCo loopback mode (same Protocol-v4 wire format, no robot). For pure server validation, run the bridge by hand with `--dry-run` to print packed action diagnostics instead of publishing ZMQ.

## ❤️ Acknowledgment

We sincerely thank [Psi0](https://github.com/physical-superintelligence-lab/Psi0), [LaST​0​](https://github.com/ZhuoyangLiu2005/last0), [GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl), and [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) for their excellent code implementations.

## 📜 Citation

```bibtex
```