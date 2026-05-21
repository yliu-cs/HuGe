# $\kappa_{0}$: A Foundation Vision-Language-Action Model for General Humanoid Robot Loco-Manipulation Control

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

## 🎯 Evaluation

```bash
uv run python scripts/eval/sim_validate.py \
  --dataset proprietary \
  --checkpoint .runs/kappa/<exp>/checkpoints/ckpt_5000/model.safetensors \
  --heads sonic gt_sonic \
  --episode-indices 0 \
  --out-dir .runs/eval/$(date +%Y%m%d_%H%M%S)
```

## 🤖 Deployment

One-time setup:

```bash
cd GR00T-WholeBodyControl
bash install_scripts/install_mujoco_sim.sh
bash install_scripts/install_data_collection.sh
cd ..
```

On the G1, launch the camera service:

```bash
sudo systemctl restart composed_camera_server.service
```

### Sim evaluation (real camera → MuJoCo)

Kappa consumes real G1 camera frames; action chunks are executed in MuJoCo and the rendered ego-view is republished over ZMQ.

```bash
./scripts/eval/run_kappa_sim_test.sh
tmux attach -t kappa_sim_test
```

Five tiled panes (`serve`, `sim`, `wbc`, `bridge`, `kbd`); `kbd` is focused. Logs land in `.runs/sim_test/<timestamp>/`. Set `CAMERA_HOST=localhost` to use MuJoCo's rendered camera for offline smoke tests.

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
