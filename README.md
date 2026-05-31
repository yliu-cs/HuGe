# Kappa

QwenSonicFM vision-language-action training stack for humanoid loco-manipulation,
targeting the 66-D SONIC action schema and the 34-D `unitree_g1_sonic v1`
proprioception layout.

Staged training path:

- Stage 1: EgoDex visual-language pretraining (64-D SONIC motion tokens).
- Stage 2: Humanoid-Everyday + G1-WBT whole-body posttraining.
- Stage 3: proprietary real-robot adaptation.

## Environment

```bash
uv sync
```

Pins CUDA 12.8 PyTorch wheels and a prebuilt `flash-attn` (Python 3.12, PyTorch
2.7, CUDA 12). Verify:

```bash
uv run python - <<'PY'
import importlib.metadata as md
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("flash-attn", md.version("flash-attn"))
PY
```

## Resources

```text
resource/
  dataset/
    EgoDex -> external EgoDex dataset
      egodex_repacked_simple/
    humanoid-everyday -> external Humanoid-Everyday dataset
    humanoid-everyday-cache/
      sonic_tokens/
      g1_proprio/
      meta/
    G1_WBT -> external G1-WBT dataset
      sonic_tokens/
      g1_proprio/
      meta/
    G1_WBT-cache/
      meta/
    pico_data -> external collected sample data
    proprietary/ego/2026-5-23/
    proprietary/real/cola_merged_cleaned_0530/
      sonic_tokens/
      g1_proprio/
      videos/
      meta/
    proprietary/simple/2026-05-21/
  model/
    Qwen3-VL-2B-Instruct/
    sonic/
```

Default Qwen3-VL path: `resource/model/Qwen3-VL-2B-Instruct`. Default SONIC
assets: `resource/model/sonic`. Download with `resource/model/hfd.sh`:

```bash
cd resource/model
./hfd.sh Qwen/Qwen3-VL-2B-Instruct --local-dir Qwen3-VL-2B-Instruct
./hfd.sh nvidia/GEAR-SONIC --local-dir sonic --include model_encoder.onnx model_decoder.onnx observation_config.yaml
```

## Data Checks

Run the audit before training:

```bash
uv run python src/kappa/data/audit.py
# equivalent launcher shortcut:
scripts/train/run_kappa.sh audit
```

It checks source existence, token/proprio shapes, hand-bit validity, and proprio
normalizer round trips.

Regenerate proprio stats if needed:

```bash
uv run python scripts/preprocess/calc_g1_proprio_stats.py
uv run python scripts/preprocess/calc_union_proprio_stats.py
```

## Training

`scripts/train/run_kappa.sh` drives stage training and the data audit. Pass the
stage explicitly (default `stage2`). The launcher auto-detects node count and
visible GPUs from `KAPPA_*`, `PET_*`, `SLURM_*`, `CUDA_VISIBLE_DEVICES`, and
`nvidia-smi` (8 bf16 processes on one 8-GPU H200 node).

```bash
# Stage 1: EgoDex pretraining
scripts/train/run_kappa.sh stage1

# Stage 2: HE + WBT posttraining
scripts/train/run_kappa.sh stage2

# Stage 3: proprietary real adaptation
scripts/train/run_kappa.sh stage3
```

`stage2`/`stage3` auto-init from the latest checkpoint of the previous stage
under `.runs/train`. Override with `--init-ckpt <path>`; use `--init-ckpt none`
to intentionally train a later stage from scratch.

Explicit distributed overrides:

```bash
scripts/train/run_kappa.sh stage2 \
  --num_gpus 8 \
  --num_nodes 1 \
  --node_rank 0 \
  --master_addr 127.0.0.1 \
  --master_port 29500
```

For multi-node jobs, set scheduler variables (`PET_NNODES`, `PET_NODE_RANK`,
`MASTER_ADDR`, `MASTER_PORT`) or the `KAPPA_*` equivalents on each node.

Checkpoints are written under
`.runs/train/<stage>/<timestamp>/step_<global_step>` as `model.safetensors` plus
JSON sidecars. Training runs no validation; use offline evaluation for visual
fit inspection.

## Stage Defaults

Current code defaults from `src/kappa/training/runner.py`:

| Stage | Purpose | Qwen3-VL strategy | LR | Per-process batch | Epochs | Warmup ratio | Weight decay | Sample stride | Max text tokens | Checkpointing |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `stage1` | EgoDex visual-language pretraining; hand dims masked | Qwen frozen, LoRA trainable | `1e-4` | `96` | `1` | `0.05` | `0.01` | `5` | `48` | every `10000` steps |
| `stage2` | Humanoid-Everyday + G1-WBT posttraining | Qwen frozen, LoRA trainable | `1e-4` | `96` | `1` | `0.05` | `0.01` | `2` | `48` | every `1000` steps |
| `stage3` | proprietary real adaptation | Qwen frozen, LoRA frozen | `2e-5` | `32` | `10` | `0.1` | `0.005` | `1` | `48` | every `2000` steps |

Shared defaults: PI model only; `action_hidden_size=1024`,
`num_action_layers=16`, `num_attention_heads=16`,
`state_dropout_prob=0.2`, `action_horizon=50`, `num_inference_steps=4`,
`gradient_accumulation_steps=1`, `num_workers=8`, `block_shuffle_size=64`,
`frame_cache_size=2`, `prefetch_factor=2`, `drop_last=false`,
`model_max_length=8192`.

Qwen hidden states are projected into the action-head width when needed, so the
action head is decoupled from the Qwen hidden size. PI uses one flow-matching
loss over the available SONIC mask. For 66-D datasets this supervises all 66
action dimensions, including hand bits; EgoDex 64-D inputs pad and mask missing
hand dims. There is no separate hand BCE branch.
Stage 3 defaults to
`resource/dataset/proprietary/real/cola_merged_cleaned_0530/meta/proprio_stats_union_sonic.json`;
Stage 2 to `resource/dataset/G1_WBT-cache/meta/g1_proprio_stats_sonic.json`.
Training images use the explicit GR00T-style preprocessing path:
pad-to-square, resize to `256x256`, train-time random `0.95` crop plus color
jitter, and eval-time center `0.95` crop without jitter.

Common overrides:

```bash
scripts/train/run_kappa.sh stage2 --batch-size 64 --learning-rate 2e-5
scripts/train/run_kappa.sh stage3 --prop-roots resource/dataset/proprietary/real/cola_merged_cleaned_0530
scripts/train/run_kappa.sh stage1
```

Explicit `--action-hidden-size`, `--num-action-layers`, and
`--num-attention-heads` overrides are accepted. The action hidden size must be
divisible by the head count, and `num_action_layers` must not exceed the loaded
Qwen layer count.

LoRA targets Qwen attention projections (`q_proj`, `k_proj`, `v_proj`,
`o_proj`) with `r=8`, `alpha=16`, dropout `0.0`. Stage 1 trains it, Stage 2
continues it, Stage 3 freezes Qwen and LoRA while training the Kappa PI
projectors, state encoder, and action head. Override with `--qwen-lora-r`,
`--qwen-lora-alpha`, `--qwen-lora-dropout`.

DeepSpeed is on by default using `configs/deepspeed/ds_zero2.json` unless
`--deepspeed-config` or `KAPPA_DEEPSPEED_CONFIG` is set. `ds_zero3.json` is
available for ZeRO-3 runs.

`--max-text-tokens` caps the raw instruction before the chat template;
`--model-max-length` is a separate full-sequence cap after image and
chat-template tokens (default `8192`). Kappa training requires Qwen processor
inputs.

## Smoke Checks

Full-flow one-step smoke on a training node:

```bash
uv lock
uv run python src/kappa/data/audit.py
scripts/train/run_kappa.sh stage1 --max-steps 1 --max-episodes 1 --batch-size 1
scripts/train/run_kappa.sh stage2 --max-steps 1 --max-episodes 1 --batch-size 1
scripts/train/run_kappa.sh stage3 --max-steps 1 --max-episodes 1 --batch-size 1
```

Local single-GPU smoke:

```bash
KAPPA_NUM_PROCESSES=1 KAPPA_EXPECTED_NUM_PROCESSES=1 \
scripts/train/run_kappa.sh stage1 \
  --max-steps 1 \
  --max-episodes 1 \
  --batch-size 1 \
  --num-workers 0
```

## Offline Evaluation

Visualize predictions vs ground truth for one or more checkpoints:

```bash
uv run python scripts/eval/offline_compare.py \
  --ckpt .runs/train/stage3/20260531_075320/step_* \
  --stage stage3 \
  --num-inference-steps 4 \
  --batch-size 8 \
  --sample-stride 1 \
  --chunk-keep-steps 50 \
  --episode-idx 123 \
  --out-dir .runs/eval/offline_compare
```

Writes one PNG per checkpoint (all SONIC dimensions and hand bits). No metrics —
visualization only.

`--chunk-keep-steps N` still asks the model to predict the full
`action_horizon`, but only stitches the first `N` actions from each predicted
chunk into the plot. With `action_horizon=50` and `N=10`, plot windows start at
`t=0,10,20,...`; each window discards predicted actions `10:50`, so the plotted
trajectory is non-overlapping and gap-free.

## Inference Server

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

uv run python src/kappa/inference/run_kappa_server.py \
  --ckpt .runs/train/stage2/<run>/step_<global_step> \
  --host 0.0.0.0 \
  --port 5550 \
  --device cuda \
  --action-horizon 50
```

Exposes a ZMQ request/reply endpoint on `tcp://<host>:<port>`. Use
`--closed-loop-enable` to also publish SONIC actions on `--sonic-pub-bind`.

## Deployment Notes

Deployment uses `GR00T-WholeBodyControl` as the robot-side runtime. Keep the
Kappa server running, then launch the deploy stack and inference client from the
WBC environment. The observation path expects the 34-D G1 SONIC proprio layout
and the proprio normalizer copied into each checkpoint directory.

```bash
# Terminal 1: Kappa server
uv run python src/kappa/inference/run_kappa_server.py \
  --ckpt .runs/train/stage3/<run>/step_<global_step> \
  --host 0.0.0.0 \
  --port 5550 \
  --device cuda

# Terminal 2: GR00T WBC deploy stack
cd GR00T-WholeBodyControl/gear_sonic_deploy
./deploy.sh --input-type zmq_manager --zmq-host localhost --hand-type inspire real

# Terminal 3: VLA inference client
cd GR00T-WholeBodyControl
python gear_sonic/scripts/run_vla_inference.py \
  --host localhost \
  --port 5550 \
  --embodiment-tag NEW_EMBODIMENT \
  --prompt "Grab the cola, turn left, throw into the trash bin" \
  --action-publish-rate 50 \
  --action-horizon 50
```

## Acknowledgment

Kappa builds on Psi0, GR00T-WholeBodyControl, Isaac-GR00T, GEAR-SONIC, and
Qwen3-VL.
