# $\kappa_{0}$: A Foundation Vision-Language-Action Model for General Humanoid Robot Loco-Manipulation Control

## TODO

- [] Adapting RTC

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
# Stage 1-1: Pre-Train on EgoDex
uv run python scripts/preprocess/egodex/build_index.py
tmux new -s pretrain_egodex -d "./scripts/train/pretrain_kappa.sh --stage egodex 2>&1 | tee pretrain_egodex.log"

# Stage 1-2: Pre-Train on Humanoid-Everyday G1 + G1-WBT
uv run python scripts/preprocess/humanoid-everyday/calc_hand_thresholds.py
tmux new -s calc_sonic -d "uv run python scripts/preprocess/humanoid-everyday/calc_sonic_tokens.py"
tmux new -s calc_wbt_sonic -d "uv run python scripts/preprocess/g1-wbt/calc_wbt_sonic_tokens.py"
tmux new -s pretrain_he_wbt -d "./scripts/train/pretrain_kappa.sh --stage he_wbt --train.pretrained-weights .runs/kappa/exp.kappa.b1024.gpus8.2604301611 | tee pretrain_he_wbt.log"

# Stage 2: Post-Train on Humanoid-Everyday G1 + G1-WBT
tmux new -s posttrain_he_wbt -d "./scripts/train/posttrain_kappa.sh | tee posttrain_he_wbt.log"

# Stage 3: Fine-Tune on Proprietary Data
tmux new -s finetune_prop -d "./scripts/train/finetune_kappa.sh | tee finetune_prop.log"
```

## Evaluation

```
MUJOCO_GL=egl uv run python scripts/eval/sim_validate.py --num-episodes 1 --out-dir .runs/eval/$(date +%Y%m%d_%H%M%S)
```

## ❤️ Acknowledgment

We sincerely thank [Psi0](https://github.com/physical-superintelligence-lab/Psi0), and [LaST​$_0$​](https://github.com/ZhuoyangLiu2005/last0) for their excellent code implementations.

## 📜 Citation

```bibtex
```