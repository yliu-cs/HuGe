# $\kappa_{0}$: A Foundation Vision-Language-Action Model for General Humanoid Robot Loco-Manipulation Control

## Overview

Kappa is a foundation vision-language-action model for humanoid loco-manipulation.
This repository provides the code and commands for environment setup, dataset/model download, staged training, offline evaluation, and real-world deployment on the G1 stack.

## 🏠 Installation

Clone the repository, enter it, and install the Python environment:

```bash
git lfs install
git clone --recurse-submodules https://github.com/yliu-cs/Kappa.git
cd Kappa
uv sync
```

### Datasets

Download the datasets used by the training pipeline into `resource/dataset/`:

```bash
wget https://hf-mirror.com/hfd/hfd.sh && chmod a+x hfd.sh
./hfd.sh USC-PSI-Lab/humanoid-everyday --dataset
# EgoDex: https://github.com/apple/ml-egodex  →  resource/dataset/EgoDex
# G1-WBT: https://huggingface.co/collections/unitreerobotics/unifolm-wbt-dataset  →  resource/dataset/G1_WBT
```

### Models

Download the base VLM and SONIC assets into `resource/model/`:

```bash
./hfd.sh Qwen/Qwen3-VL-2B-Instruct
./hfd.sh nvidia/GEAR-SONIC --local-dir ./sonic --include model_encoder.onnx model_decoder.onnx observation_config.yaml
```

## 🚀 Training

Run the staged training pipeline below. The inline comments mark the three training stages and the final end-to-end shortcut:

```bash
# Stage 1: Pre-Train on EgoDex
uv run python scripts/preprocess/egodex/build_index.py
./scripts/train/train_kappa.sh pretrain_egodex_kappa_config

# Stage 2: Post-Train on Humanoid-Everyday G1 + G1-WBT
uv run python scripts/preprocess/humanoid-everyday/calc_hand_thresholds.py
uv run python scripts/preprocess/humanoid-everyday/calc_sonic_tokens.py
uv run python scripts/preprocess/g1-wbt/calc_wbt_sonic_tokens.py
./scripts/train/train_kappa.sh posttrain_he_wbt_kappa_config

# Stage 3: Fine-Tune on Proprietary Data
uv run python scripts/preprocess/proprietary/calc_sonic_tokens.py
./scripts/train/train_kappa.sh finetune_prop_ep_kappa_config

# Or run all three stages end-to-end:
./scripts/train/run_kappa.sh
```

## 🧪 Offline Evaluation

Use the offline evaluator to compare Kappa predictions against ground-truth SONIC tokens before deployment. It tiles one recorded episode with action-horizon windows, runs one or more checkpoints with the same flow-matching sampler used by the deployment server, and writes one full-episode PNG per checkpoint for visual inspection.

```bash
uv run python scripts/eval/offline_compare.py \
  --ckpt-path .runs/kappa/exp.kappa.b1024.gpus8.2605280508/checkpoints/ckpt_* \
  --data-root resource/dataset/proprietary/real/2026-05-23 \
  --base-vlm-path resource/model/Qwen3-VL-2B-Instruct \
  --episode-idx 123 \
  --action-horizon 50 \
  --num-inference-steps 4 \
  --out-dir .runs/eval/offline_compare/ep123
```

`--ckpt-path` accepts one or more explicit paths, `ckpt_*` directories, or glob patterns that expand to multiple checkpoints.

## 🤖 Deployment

### Launch G1

Connect to the robot host and restart the camera and control services required for deployment:

```bash
ssh unitree@192.168.123.164
v4l2-ctl --list-devices
sudo systemctl restart composed_camera_server.service
sudo systemctl restart inspire-g1.service
```

### Prepare Inference Pipeline

Clone and prepare the GR00T-WholeBodyControl inference stack, which provides the deployment-time runtime and helper scripts:

```bash
git clone https://github.com/Kinetra-Humanoids/GR00T-WholeBodyControl.git
cd GR00T-WholeBodyControl
git lfs pull
cd gear_sonic_deploy
chmod +x scripts/install_deps.sh
./scripts/install_deps.sh
source scripts/setup_env.sh
cd ..
pip install huggingface_hub
python download_from_hf.py
bash install_scripts/install_mujoco_sim.sh
bash install_scripts/install_data_collection.sh
bash install_scripts/install_inference.sh
```

### Patch Inference Script

Patch the inference script so it forwards the quaternion and projected-gravity state expected by Kappa:

```python
# Projected gravity for Sonic latent embodiment
assert "base_quat" in state_msg, "base_quat not found in state_msg"
base_quat = np.asarray(state_msg["base_quat"], dtype=np.float64)
assert base_quat.shape == (4,), "base_quat must have shape (4,)"
projected_gravity = compute_projected_gravity(base_quat)
observation["state"]["projected_gravity"] = np.asarray(
    projected_gravity, dtype=np.float32
)[np.newaxis, np.newaxis]
# Forward the raw pelvis quaternion (WXYZ order) so policies that need the
# full orientation (e.g. Kappa's 64-D proprio `quat_xyzw` slot) can recover
# it without losing yaw to projected_gravity.
observation["state"]["base_quat"] = base_quat.astype(np.float32)[
    np.newaxis, np.newaxis
]
```

### Real World Deployment

Run the four terminals below to start the Kappa server, the deploy stack, the inference client, and the keyboard publisher:

```bash
# Terminal 1: Start Kappa server
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
uv run python scripts/eval/run_kappa_server.py \
  --ckpt-path .runs/kappa/exp.kappa.b1024.gpus8.2605280508/checkpoints/ckpt_2000 \
  --base-vlm-path resource/model/Qwen3-VL-2B-Instruct \
  --port 5550 \
  --action-horizon 50 \
  --num-inference-steps 4

# Terminal 2: Start deploy stack
cd GR00T-WholeBodyControl/gear_sonic_deploy
./deploy.sh --input-type zmq_manager --zmq-host localhost --hand-type inspire real

# Terminal 3: Run inference client
cd GR00T-WholeBodyControl
source .venv_inference/bin/activate
python gear_sonic/scripts/run_vla_inference.py \
  --host localhost \
  --port 5550 \
  --embodiment-tag NEW_EMBODIMENT \
  --prompt "Grab the cola, turn left, throw into the trash bin" \
  --action-publish-rate 50 \
  --action-horizon 50 \
  --camera-host 192.168.123.164 \
  --camera-port 5555

# Terminal 4: Start keyboard publisher
cd GR00T-WholeBodyControl
source .venv_inference/bin/activate
python keyboard_pub.py
```

## ❤️ Acknowledgment

We thank [Psi0](https://github.com/physical-superintelligence-lab/Psi0), [LaST​0​](https://github.com/ZhuoyangLiu2005/last0), [GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl), and [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) for their excellent code.

## 📜 Citation

```bibtex
```
