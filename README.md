# $\kappa_{0}$: A Foundation Vision-Language-Action Model for General Humanoid Robot Loco-Manipulation Control

## 🏠 Installation

```
git clone https://github.com/yliu-cs/Kappa.git
cd Kappa
uv sync
uv pip install flash_attn==2.7.4.post1 --no-build-isolation

mkdir -p resource/dataset
wget https://hf-mirror.com/hfd/hfd.sh && chmod a+x hfd.sh
./hfd.sh USC-PSI-Lab/humanoid-everyday --dataset
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
uv run python scripts/preprocess/humanoid-everyday/calc_hand_thresholds.py
uv run python scripts/preprocess/humanoid-everyday/calc_sonic_tokens.py

tmux new -s pretrain_kappa -d "./scripts/train/pretrain_kappa.sh 2>&1 | tee pretrain_kappa.log"
```

## ❤️ Acknowledgment

We sincerely thank [Psi0](https://github.com/physical-superintelligence-lab/Psi0), and [LaST​$_0$​](https://github.com/ZhuoyangLiu2005/last0) for their excellent code implementations.

## 📜 Citation

```bibtex
```