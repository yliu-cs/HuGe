# $\kappa_{0}$: A Foundation Vision-Language-Action Model for General Humanoid Robot Loco-Manipulation Control

claude --resume 0e4c450b-3c00-471f-b451-7c00b18ca485

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
uv run python scripts/eval/sim_validate.py --heads gt_wbc sonic --out-dir .runs/eval/$(date +%Y%m%d_%H%M%S)
```

## ❤️ Acknowledgment

We sincerely thank [Psi0](https://github.com/physical-superintelligence-lab/Psi0), and [LaST​$_0$​](https://github.com/ZhuoyangLiu2005/last0) for their excellent code implementations.

## 📜 Citation

```bibtex
```