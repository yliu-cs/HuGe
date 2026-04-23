```
git clone https://github.com/yliu-cs/Kappa.git
cd Kappa

mkdir -p resource/dataset

mkdir -p resource/model
cd resource/model
wget https://hf-mirror.com/hfd/hfd.sh && chmod a+x hfd.sh
./hfd.sh Qwen/Qwen3-VL-2B-Instruct
mkdir -p sonic
./hfd.sh nvidia/GEAR-SONIC --local-dir ./sonic --include model_encoder.onnx
./hfd.sh nvidia/GEAR-SONIC --local-dir ./sonic --include model_decoder.onnx
./hfd.sh nvidia/GEAR-SONIC --local-dir ./sonic --include observation_config.yaml
```


```
uv run python scripts/calc_sonic_tokens.py

uv run python scripts/train.py pretrain_kappa_config --train.max_training_steps=100
```