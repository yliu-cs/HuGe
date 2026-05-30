# Kappa

Kappa is a QwenSonicFM vision-language-action training stack for humanoid
loco-manipulation. This rewrite targets the 66-D SONIC action schema and the
34-D `unitree_g1_sonic v1` proprioception layout used by the G1 deployment
stack.

The training path is staged:

- Stage 1: EgoDex visual-language pretraining with 64-D SONIC motion tokens.
- Stage 2: Humanoid-Everyday + G1-WBT whole-body posttraining.
- Stage 3: proprietary real-robot adaptation with G1-WBT replay.

## Environment

Install the project environment with `uv`:

```bash
uv sync
```

The project pins the CUDA 12.8 PyTorch wheels and installs `flash-attn` from a
prebuilt wheel matched to Python 3.12, PyTorch 2.7, and CUDA 12:

```bash
uv run python - <<'PY'
import importlib.metadata as md
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("flash-attn", md.version("flash-attn"))
PY
```

## Resources

Expected resource layout:

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
    proprietary/real/2026-05-28/
      sonic_tokens/
      g1_proprio/
      videos/
      meta/
    proprietary/simple/2026-05-21/
  model/
    Qwen3-VL-2B-Instruct/
    sonic/
```

The default Qwen3-VL path is `resource/model/Qwen3-VL-2B-Instruct`. The default
SONIC assets live under `resource/model/sonic`.

Download helpers are available under `resource/model/hfd.sh`, for example:

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

The report is written to `resource/info/data_audit_report.md`. It checks source
existence, token/proprio shapes, hand-bit validity, and proprio normalizer
round trips.

Text instruction length statistics are recorded in
`resource/info/text_instruction_length_stats_unweighted.json`. The stage text
limits below are chosen from the unweighted Qwen text-token distribution, so
each instruction/episode contributes equally.

If the proprio stats need to be regenerated:

```bash
uv run python scripts/preprocess/calc_g1_proprio_stats.py
uv run python scripts/preprocess/calc_union_proprio_stats.py
```

## Training

Use `scripts/train/run_kappa.sh` for stage training and data audit. The default
stage is `stage2` if no stage argument is provided, but production jobs should
pass the stage explicitly. The launcher auto-detects node
count and visible GPUs from `KAPPA_*`, `PET_*`, `SLURM_*`,
`CUDA_VISIBLE_DEVICES`, and `nvidia-smi`. On one 8-GPU H200 node it resolves to
8 bf16 processes by default.

```bash
# Stage 1: EgoDex pretraining
scripts/train/run_kappa.sh stage1

# Stage 2: HE + WBT posttraining
scripts/train/run_kappa.sh stage2

# Stage 3: proprietary real adaptation + WBT replay
scripts/train/run_kappa.sh stage3
```

When launching `stage2` or `stage3`, the script automatically initializes from
the latest checkpoint of the previous stage under `.runs/train`:
`stage2` loads the latest `stage1/.../step_*`, and `stage3` loads the latest
`stage2/.../step_*`. Override this with `--init-ckpt <path>`; pass
`--init-ckpt none` only when intentionally training a later stage from scratch.

The launcher also accepts explicit distributed overrides, compatible with the
pre-rewrite launch style:

```bash
scripts/train/run_kappa.sh stage2 \
  --num_gpus 8 \
  --num_nodes 1 \
  --node_rank 0 \
  --master_addr 127.0.0.1 \
  --master_port 29500
```

For multi-node jobs, set the scheduler variables (`PET_NNODES`,
`PET_NODE_RANK`, `MASTER_ADDR`, `MASTER_PORT`) or the equivalent `KAPPA_*`
variables on each node before invoking the same script.

Checkpoints are written under `.runs/train/<stage>/<timestamp>/step_<global_step>`
as `model.safetensors` plus JSON sidecars.

Training does not run validation. Use the offline evaluation script after saving
a checkpoint when visual fit inspection is needed.

## Stage Defaults

These values are the current code defaults from `src/kappa/training/runner.py`.
They intentionally differ from the earlier planning notes in
`resource/info/qwen_sonicfm_execution_plan.md`.

| Stage | Purpose | Qwen3-VL strategy | LR | Per-process batch | Epochs | Sample stride | Max text tokens | Checkpointing |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `stage1` | EgoDex visual-language pretraining; hand dims masked | LoRA trainable | `1e-4` | `256` | `1` | `10` | `48` | every `30000` steps |
| `stage2` | Humanoid-Everyday + G1-WBT posttraining | frozen Qwen, LoRA modules present but frozen | `5e-5` | `256` | `1` | `2` | `48` | every `1000` steps |
| `stage3` | proprietary real adaptation + WBT replay | frozen Qwen, LoRA modules present but frozen | `3e-5` | `256` | `30` | `1` | `48` | every `600` steps and every epoch |

Additional defaults shared by all stages: `hidden_size=512`,
`vlm_hidden_size=512`, `action_horizon=50`, `num_inference_steps=4`,
`gradient_accumulation_steps=1`, `num_workers=8`, `block_shuffle_size=64`,
`frame_cache_size=2`, `prefetch_factor=2`, `drop_last=false`,
`model_max_length=8192`, and `char_max_length=256`.

Stage-specific hand loss weights are `0.0` for `stage1`, `0.5` for `stage2`,
and `1.0` for `stage3`. Stage 3 uses
`resource/dataset/proprietary/real/2026-05-28/meta/proprio_stats_union_sonic.json`
by default; Stage 2 uses
`resource/dataset/G1_WBT-cache/meta/g1_proprio_stats_sonic.json`.

Common overrides:

```bash
scripts/train/run_kappa.sh stage2 --batch-size 64 --learning-rate 2e-5
scripts/train/run_kappa.sh stage3 --prop-roots resource/dataset/proprietary/real/2026-05-28
scripts/train/run_kappa.sh stage1 --no-use-qwen --no-use-deepspeed
```

Stage 1 applies LoRA to Qwen attention projections (`q_proj`, `k_proj`,
`v_proj`, `o_proj`) with `r=8`, `alpha=16`, and dropout `0.0` by default.
Stage 2 and Stage 3 retain the Stage 1 LoRA modules but freeze them while
continuing to train the Kappa projection, state encoder, and action head.
Override the Stage 1 adapter settings with `--qwen-lora-r`,
`--qwen-lora-alpha`, and `--qwen-lora-dropout`.

DeepSpeed is enabled by default and uses `configs/deepspeed/ds_zero2.json`
unless `--deepspeed-config` or `KAPPA_DEEPSPEED_CONFIG` is set. Available
configs are `ds_zero2.json`, `ds_zero2_offload.json`, `ds_zero3.json`, and
`ds_zero3_offload.json`.

`--max-text-tokens` caps the raw task instruction before the Qwen chat template.
`--model-max-length` keeps a separate full-sequence safety cap after image
tokens and chat-template tokens are added; it defaults to `8192`.

`--no-use-qwen` is only for data/loss/checkpoint debugging. Production training
routes text/images through the local Qwen3-VL processor and model.

## Smoke Checks

For a full environment smoke on a normal training node:

```bash
uv lock
uv run python src/kappa/data/audit.py
scripts/train/run_kappa.sh stage1 --max-steps 1 --max-episodes 1 --batch-size 1
scripts/train/run_kappa.sh stage2 --max-steps 1 --max-episodes 1 --batch-size 1
scripts/train/run_kappa.sh stage3 --max-steps 1 --max-episodes 1 --batch-size 1
```

For a local single-GPU smoke, override the distributed size and optionally skip
Qwen/DeepSpeed:

```bash
KAPPA_NUM_PROCESSES=1 KAPPA_EXPECTED_NUM_PROCESSES=1 \
scripts/train/run_kappa.sh stage1 \
  --max-steps 1 \
  --max-episodes 1 \
  --num-workers 0 \
  --no-use-qwen \
  --no-use-deepspeed
```

## Offline Evaluation

Generate an offline prediction-vs-ground-truth visualization for one or more
checkpoints:

```bash
uv run python scripts/eval/offline_compare.py \
  --ckpt .runs/train/stage3/20260530_103709/step_* \
  --stage stage3 \
  --batch-size 8 \
  --sample-stride 1 \
  --episode-idx 123 \
  --out-dir .runs/eval/offline_compare
```

The evaluator writes one PNG per checkpoint under `--out-dir`. Each image shows
predicted vs ground-truth SONIC dimensions (all dimensions) and hand bits. No
metrics are computed or saved — visualization only.

## Inference Server

Start the Kappa policy server from a saved checkpoint:

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

The server exposes a ZMQ request/reply endpoint on `tcp://<host>:<port>`. Use
`--closed-loop-enable` to also publish SONIC actions on `--sonic-pub-bind`
for closed-loop integration.

## Deployment Notes

The deployment path uses `GR00T-WholeBodyControl` as the robot-side runtime.
Keep the Kappa server running, then launch the deploy stack and inference
client from the WBC environment. The Kappa observation path expects the 34-D G1
SONIC proprio layout and the same proprio normalizer copied into each
checkpoint directory.

Minimal terminal layout:

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

Kappa builds on ideas and tooling from Psi0, GR00T-WholeBodyControl,
Isaac-GR00T, GEAR-SONIC, and Qwen3-VL.
