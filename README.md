# $\kappa_{0}$: A Foundation Vision-Language-Action Model for General Humanoid Robot Loco-Manipulation Control

`claude --resume cb88f681-bf96-49eb-a4c3-8bfc479ed08e`

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
./hfd.sh nvidia/GEAR-SONIC --local-dir ./sonic --include model_encoder.onnx model_decoder.onnx observation_config.yaml
```

## 🚀 Training

```bash
# Stage 1: Pre-Train on EgoDex
uv run python scripts/preprocess/egodex/build_index.py
./scripts/train/train_kappa.sh pretrain_egodex_kappa_config

# Stage 2: Post-Train on Humanoid-Everyday G1 + G1-WBT (SONIC tokens)
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

## Deploy

```
ssh unitree@192.168.123.164
v4l2-ctl --list-devices
sudo systemctl restart composed_camera_server.service
sudo systemctl restart inspire-g1.service
```

```
# Terminal 1:
cd /home/kinetra-5090-01/code/liuyang/Kappa
source .venv/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NO_ALBUMENTATIONS_UPDATE=1
python scripts/eval/run_kappa_server.py \
--ckpt-path .runs/kappa/exp.kappa.b1024.gpus8.2605262100/checkpoints/ckpt_3000 \
--base-vlm-path resource/model/Qwen3-VL-2B-Instruct \
--port 5550 \
--action-horizon 50 \
--num-inference-steps 4

# Terminal 2:
cd /home/kinetra-5090-01/code/liuyang/Kappa/GR00T-WholeBodyControl/gear_sonic_deploy
./deploy.sh --input-type zmq_manager --zmq-host localhost --hand-type inspire real

# Terminal 3:
cd /home/kinetra-5090-01/code/liuyang/Kappa/GR00T-WholeBodyControl
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

# Terminal 4:
cd /home/kinetra-5090-01/code/liuyang/Kappa/GR00T-WholeBodyControl
source .venv_inference/bin/activate
python keyboard_pub.py
```

Terminal 1 records every ZMQ request/response under `.runs/deploy_log/<stamp>/`
(one msgpack file per call + a `manifest.jsonl` summary), capturing the full
observation, the assembled proprio (pre- and post-normalization), and the
returned action chunks. Each `get_action` call also dumps the first camera
frame to `calls/call_<idx>.jpg` for quick visual inspection. Use this log to
compare deploy-time inputs/outputs against the training set when diagnosing
failures. Disable with `--no-log-enable` (or `--no-log-images` to keep
msgpack but skip the JPEGs).

## ❤️ Acknowledgment

We thank [Psi0](https://github.com/physical-superintelligence-lab/Psi0), [LaST​0​](https://github.com/ZhuoyangLiu2005/last0), [GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl), and [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) for their excellent code.

## 📜 Citation

```bibtex
```
