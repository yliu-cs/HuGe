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
./hfd.sh nvidia/GEAR-SONIC --local-dir ./sonic \
  --include model_encoder.onnx model_decoder.onnx observation_config.yaml
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

## ❤️ Acknowledgment

We thank [Psi0](https://github.com/physical-superintelligence-lab/Psi0), [LaST​0​](https://github.com/ZhuoyangLiu2005/last0), [GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl), and [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) for their excellent code.

## 📜 Citation

```bibtex
```
