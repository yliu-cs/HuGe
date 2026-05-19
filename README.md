# $\kappa_{0}$: A Foundation Vision-Language-Action Model for General Humanoid Robot Loco-Manipulation Control

claude --resume 31ae9900-a7da-415e-8a42-71657ececc36

## 🏠 Installation

```
git clone https://github.com/yliu-cs/Kappa.git
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

## ❤️ Acknowledgment

We sincerely thank [Psi0](https://github.com/physical-superintelligence-lab/Psi0), and [LaST​$_0$​](https://github.com/ZhuoyangLiu2005/last0) for their excellent code implementations.

## 📜 Citation

```bibtex
```