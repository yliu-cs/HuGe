# $\kappa_{0}$: A Foundation Vision-Language-Action Model for General Humanoid Robot Loco-Manipulation Control

`claude --resume 28545022-29bc-4a84-92dc-84e40fa7f9ca`

## 🏠 Installation

```bash
git lfs install
git clone --recurse-submodules https://github.com/yliu-cs/Kappa.git
cd Kappa
uv sync
```

Download datasets to `resource/dataset/`:

```bash
wget https://hf-mirror.com/hfd/hfd.sh && chmod a+x hfd.sh
./hfd.sh USC-PSI-Lab/humanoid-everyday --dataset
# EgoDex: https://github.com/apple/ml-egodex  →  resource/dataset/EgoDex
# G1-WBT: https://huggingface.co/collections/unitreerobotics/unifolm-wbt-dataset  →  resource/dataset/G1_WBT
```

Download models to `resource/model/`:

```bash
./hfd.sh Qwen/Qwen3-VL-2B-Instruct
./hfd.sh nvidia/GEAR-SONIC --local-dir ./sonic \
  --include model_encoder.onnx model_decoder.onnx observation_config.yaml
```

## 🚀 Training

```bash
# Stage 1: Pre-Train on EgoDex
uv run python scripts/preprocess/egodex/build_index.py
./scripts/train/train_kappa.sh pretrain_egodex_kappa_config

# Stage 2: Post-Train on Humanoid-Everyday G1 + G1-WBT (SONIC tokens + RTC)
uv run python scripts/preprocess/humanoid-everyday/calc_hand_thresholds.py
uv run python scripts/preprocess/humanoid-everyday/calc_sonic_tokens.py
uv run python scripts/preprocess/g1-wbt/calc_wbt_sonic_tokens.py
./scripts/train/train_kappa.sh posttrain_he_wbt_kappa_config

# Stage 3: Fine-Tune on Proprietary Data (SONIC tokens)
uv run python scripts/preprocess/proprietary/calc_sonic_tokens.py
./scripts/train/train_kappa.sh finetune_prop_ep_kappa_config

# Or run all three stages end-to-end:
./scripts/train/run_kappa.sh
```

## 🎯 Offline Closed-Loop Sim

`sim_validate.py` is the offline closed-loop simulation entry point. It keeps
the deployment Kappa server, WebSocket bridge, WBC deploy binary, and GR00T
MuJoCo sim loop in the path. The model input is published from the proprietary
offline cache: dataset `ego_view` frames on the camera ZMQ port and cached
64-D proprio converted to the bridge's `g1_debug` state format. Kappa actions
are then executed through the real bridge/WBC/sim loop, so this tests whether
the deployed chain can drive the MuJoCo G1 stably from offline inputs.

```bash
.venv/bin/python scripts/eval/sim_validate.py \
  --dataset proprietary \
  --episode-index 0 \
  --checkpoint .runs/kappa/exp.kappa.b1024.gpus8.2605181229/checkpoints/ckpt_5000/model.safetensors \
  --backbone resource/model/Qwen3-VL-2B-Instruct \
  --proprio-stats resource/dataset/G1_WBT-cache/meta/g1_proprio_stats.json \
  --device cuda:0 \
  --ctrl-hz 50 \
  --session kappa_sim_test
```

Attach with `tmux attach -t kappa_sim_test`. Logs land in
`.runs/sim_test/<timestamp>/`.

## ⏱️ Kappa Inference Speed Benchmark

Use this script to benchmark the Kappa model inference path used by
`scripts/eval/run_kappa_sim_test.sh` without launching WebSocket, bridge, WBC,
or MuJoCo. It reuses the server-side model loading, image preprocessing,
proprio normalization, scheduler setup, and RTC chunk inference call.

```bash
.venv/bin/python scripts/eval/benchmark_kappa_inference_speed.py \
  --weights .runs/kappa/170R+85E/real/checkpoints/ckpt_1300/model.safetensors \
  --backbone resource/model/Qwen3-VL-2B-Instruct \
  --proprio-stats resource/dataset/G1_WBT-cache/meta/g1_proprio_stats.json \
  --device cuda:0 \
  --input-source offline \
  --data-root resource/dataset/proprietary/real/2026-05-23 \
  --episode-index 123 \
  --warmup 3 \
  --iters 30 \
  --json-out .runs/eval/kappa_infer_speed.json
```

For a quick model-path smoke benchmark without reading the offline dataset:

```bash
.venv/bin/python scripts/eval/benchmark_kappa_inference_speed.py \
  --input-source dummy \
  --weights .runs/kappa/170R+85E/real/checkpoints/ckpt_1300/model.safetensors
```

## 🤖 Deployment

One-time setup:

```bash
cd GR00T-WholeBodyControl
bash install_scripts/install_mujoco_sim.sh
bash install_scripts/install_data_collection.sh
cd ..
```

SSH to G1, launch the camera service:

```bash
ssh root@192.168.123.164
v4l2-ctl --list-devices
sudo systemctl restart composed_camera_server.service
sudo systemctl restart inspire-g1.service
```

Connect to the camera streaming:

```bash
cd GR00T-WholeBodyControl
source .venv_data_collection/bin/activate
python gear_sonic/scripts/run_camera_viewer.py \
  --camera-host 192.168.123.164 \
  --camera-port 5555
```

### Sim evaluation

By default this is the offline-cache closed-loop sim described above.

```bash
.venv/bin/python scripts/eval/sim_validate.py
tmux attach -t kappa_sim_test
```

Tiled panes are `serve`, `sim`, `wbc`, `cache`, `bridge`, and `kbd`; `kbd` is
focused. Set `OFFLINE_CACHE_INPUT=0` only when you intentionally want the
bridge to read camera/state from external publishers.

### Real-world evaluation

The helper runs pre-flight checks, reclaims stale ports, then prints four commands to paste into four terminals:

```bash
./scripts/deploy/run_kappa_deploy.sh
```

Start order:

1. **serve** — wait for `Listening on $KAPPA_HOST:$WS_PORT`.
2. **wbc** — answer `y`; wait for `Init Done` plus ~5s for the torso-IMU DDS handshake.
3. **bridge** — start only after `serve` is listening.
4. **kbd** — press `k` (engage motors, PLANNER mode), `i` (initial motion token, POSE mode), `p` (unpause bridge).

**Safety**: position the robot and hold the e-stop before pressing `k`.

## ❤️ Acknowledgment

We thank [Psi0](https://github.com/physical-superintelligence-lab/Psi0), [LaST​0​](https://github.com/ZhuoyangLiu2005/last0), [GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl), and [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) for their excellent code.

## 📜 Citation

```bibtex
```
