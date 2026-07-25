# 🤖 Huge: Humanoid General Foundation Model

Huge is a Qwen3-VL vision-language-action training and inference stack for
humanoid loco-manipulation.

## 🧩 Core contracts

| Item | Value |
| --- | --- |
| Action | 66-D: 64-D SONIC motion token + 2 binary hand states |
| Hand state | left/right; closed = 1, open = 0 |
| Proprioception | 34-D `unitree_g1_sonic v1` |
| Action horizon | 50 |
| Control rate | 50 Hz |
| Architectures | `qwenpi_v3`, `qwengr00t`, `qwenoft` |
| Base VLM | `resource/model/Qwen3-VL-2B-Instruct` |
| SONIC assets | `resource/model/sonic` |

Training stages:

| Stage | Data |
| --- | --- |
| `stage1` | EgoDex, EgoVerse, Behavior-1K, and Xperience adapters |
| `stage2` | Humanoid-Everyday, G1-WBT, and HIW-500 |
| `stage3` | proprietary real/sim data or the unified HumanoidArena cache |

## ⚙️ Requirements

- Linux x86_64
- Python 3.12
- `uv`
- CUDA 12.8-compatible driver and GPU for training

The lock file pins PyTorch 2.7.1 CUDA 12.8 wheels and the Python 3.12
`flash-attn` 2.8.3 wheel.

Run all commands from the repository root.

```bash
uv sync --frozen
```

Verify the environment:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python - <<'PY'
import importlib.metadata as md
import torch
import huge

print("huge:", huge.__file__)
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("flash-attn:", md.version("flash-attn"))
PY
```

Use `PYTHONPATH=src .venv/bin/python` for Huge Python entry points. Do not
mix the root environment with the separate WBC or Isaac Lab environments.

## 🗂️ Repository layout

```text
src/huge/
  data/          datasets, schemas, preprocessing, stats, normalization
  models/        QwenSonicFM and action heads
  training/      runner, losses, checkpointing
  inference/     policy, ZMQ server, SONIC adapter, deployment logging
scripts/
  preprocess/
  train/
  eval/
configs/deepspeed/
tests/
resource/
  datasets/
  model/
.runs/
GR00T-WholeBodyControl/
HumanoidArena/
```

`GR00T-WholeBodyControl` and `HumanoidArena` are external runtime projects
with their own environments and instructions.

## 📦 Model assets

Download the local Qwen3-VL model and SONIC ONNX assets:

```bash
resource/model/hfd.sh Qwen/Qwen3-VL-2B-Instruct \
  --tool wget \
  --local-dir resource/model/Qwen3-VL-2B-Instruct

resource/model/hfd.sh nvidia/GEAR-SONIC \
  --tool wget \
  --local-dir resource/model/sonic \
  --include model_encoder.onnx model_decoder.onnx observation_config.yaml
```

The Qwen loader uses local files only.

## 🗃️ HumanoidArena data

### Required inputs

```text
resource/datasets/
  HumanoidArena_raw/
  humanoid-everyday-cache/
    g1_proprio/
  G1_WBT/
    g1_proprio/
```

`G1_WBT` may be an external symlink.

### Download raw SONIC data

The following command downloads every top-level task's `sonic/` or
`SONIC/` subtree from `Twang2026/HumanoidArena_raw@master`:

```bash
PYTHONPATH=src .venv/bin/python - <<'PY'
from modelscope.hub.snapshot_download import dataset_snapshot_download

dataset_snapshot_download(
    dataset_id="Twang2026/HumanoidArena_raw",
    revision="master",
    local_dir="resource/datasets/HumanoidArena_raw",
    allow_patterns=["*/sonic/**", "*/SONIC/**"],
)
PY
```

This is a network and disk-intensive operation.

### Build the unified cache

```bash
PYTHONPATH=src .venv/bin/python \
  scripts/preprocess/proprietary/calc_sonic_tokens_from_humanoid_arena.py
```

Default inputs and output:

| Argument | Default |
| --- | --- |
| `--raw-root` | `resource/datasets/HumanoidArena_raw` |
| `--out-root` | `resource/datasets/HumanoidArena` |
| `--he-cache-root` | `resource/datasets/humanoid-everyday-cache` |
| `--wbt-source-root` | `resource/datasets/G1_WBT` |

The preprocessor validates all episodes, writes to staging, computes
HE + WBT + HumanoidArena union stats, validates the cache, and publishes it
atomically.

```text
resource/datasets/HumanoidArena/
  _SUCCESS
  sonic_tokens/
  g1_proprio/
  videos/chunk-000/observation.images.ego_view/
  meta/
    conversion_manifest.json
    language_map.json
    proprio_stats_union_sonic.json
```

Embedded v2 RGB frames are encoded to MP4. V3 sidecar videos are linked back to
the raw tree. Do not move or delete `HumanoidArena_raw` after preprocessing.

An unchanged completed cache is reused after validation. Use `--rebuild` only
when intentionally replacing an existing cache.

## 🏋️ Training

Training must use the launcher:

```bash
bash scripts/train/train_huge.sh <stage> [options]
```

The launcher configures Accelerate, DeepSpeed, GPU/process discovery, cuDNN,
and previous-stage checkpoint selection.

### Stage 1/2 multi-source data

Repeated source arguments provide the common upper-level interface; each kind
uses its own lower-level adapter. Omitting them preserves the legacy defaults:
Stage 1 uses EgoDex only and Stage 2 uses HE + WBT.

```bash
bash scripts/train/train_huge.sh stage1 \
  --stage1-source egodex resource/datasets/EgoDex \
  --stage1-source egoverse resource/datasets/egoverse-cache \
  --stage1-source behavior1k resource/datasets/behavior-1k-2025-challenge-cache \
  --stage1-source xperience resource/datasets/xperience-10m-cache \
  --episode-fraction 0.10 \
  --episode-selection-seed 7 \
  --sample-stride 10

bash scripts/train/train_huge.sh stage2 \
  --stage2-source he resource/datasets/humanoid-everyday \
  --stage2-source wbt resource/datasets/G1_WBT \
  --stage2-source hiw resource/datasets/HIW-500 \
  --proprio-stats-path \
    resource/datasets/G1_WBT-cache/meta/g1_proprio_stats_sonic.json
```

The full-pipeline launcher now defaults Stage 1 to EgoDex + EgoVerse +
BEHAVIOR-1K + Xperience-10M and Stage 2 to Humanoid Everyday + G1_WBT +
HIW-500. Stage 1 fractions select a deterministic, nested subset of complete
source episodes; `sample_stride` is a deterministic within-episode time-step
stride, not random episode sampling. `--max-episodes` remains a mutually
exclusive per-source prefix limit for debugging. Sources are mixed in canonical
order at their natural sample ratio.

The HIW adapter reads `HIW-500/meta/episodes` directly; when that directory is
absent, it downloads only the official `BitRobot/HIW-500-LeRobot` episode
metadata into the standard Hugging Face cache. Publish the derived 34-D state
after the in-place 66-D token upgrade:

```bash
PYTHONPATH=src .venv/bin/python \
  scripts/preprocess/hiw/prepare_hiw_cache.py --publish-source-proprio
```

The existing `resource/datasets/HIW-500-cache` still contains only `_SUCCESS`
and a validation manifest. Derived arrays live beside the source as
`HIW-500/sonic_tokens/` and `HIW-500/g1_proprio/`; videos and Parquet files are
not copied. HIW proprio uses the source 29 G1 joint positions, event-latched WBC
hand state, and projected gravity derived from WBC pivot roll/pitch/yaw. Stage 2
and Stage 3 union statistics include those valid HIW states. Head AV1 video is
decoded strictly through PyAV/libdav1d.


### HumanoidArena single configuration

```bash
bash scripts/train/train_huge.sh stage3 \
  --model-architecture qwenpi_v3 \
  --init-ckpt none \
  --prop-roots \
  --humanoid-arena-roots resource/datasets/HumanoidArena \
  --humanoid-arena-task HOI_pp_box \
  --proprio-stats-path \
    resource/datasets/HumanoidArena/meta/proprio_stats_union_sonic.json
```

The empty `--prop-roots` is intentional; it disables the default proprietary
dataset.

Use `--humanoid-arena-task all` for the full unified cache. Valid task names
are recorded in `meta/conversion_manifest.json`.

### HumanoidArena experiment matrix

`scripts/train/run_huge.sh` defaults to three sequential Stage 3 full-tuning runs:

- task: `HOI_pp_box`
- architectures: `qwenpi_v3 qwengr00t qwenoft`
- tuning modes: `full lora frozen_vlm`
- initialization: none
- epochs: 50
- checkpoint interval: 3000 steps

Tuning modes:

| Mode | VLM update |
| --- | --- |
| `full` | train the full Qwen model |
| `lora` | train attention LoRA; default r=8, alpha=16, dropout=0 |
| `frozen_vlm` | freeze Qwen; train Huge action components |

Preview one configuration without training:

```bash
tmp_dir="$(mktemp -d /tmp/huge-matrix-dry-run.XXXXXX)"

HUGE_DRY_RUN=1 \
HUGE_STAGE3_DATA=humanoid_arena \
HUGE_HUMANOID_ARENA_ARCHITECTURES=qwenpi_v3 \
HUGE_HUMANOID_ARENA_TUNING_MODES=full \
HUGE_LOCK_FILE="$tmp_dir/run.lock" \
HUGE_RUN_ROOT="$tmp_dir/runs" \
bash scripts/train/run_huge.sh
```

Run the default matrix:

```bash
bash scripts/train/run_huge.sh
```

Run one selected configuration:

```bash
HUGE_HUMANOID_ARENA_TASK=HOI_pp_box \
HUGE_HUMANOID_ARENA_ARCHITECTURES=qwenpi_v3 \
HUGE_HUMANOID_ARENA_TUNING_MODES=full \
bash scripts/train/run_huge.sh
```

Legacy pipeline modes:

- `HUGE_STAGE3_DATA=sim` runs Stage 1 -> Stage 2 -> sim Stage 3 once over every
  EgoDex time step and writes under `.runs/sim_pipeline`.
- `HUGE_STAGE3_DATA=real` runs real Stage 3 once, initializes from
  `HUGE_STAGE2_SOURCE_ROOT`, and writes under `.runs/stage3_real_cola`.

Both modes retain legacy `resource/dataset/` defaults and require path
alignment before use.

Outputs:

```text
.runs/stage3_humanoidarena/
  <task>/<architecture>/<tuning-mode>/stage3/<run-id>/
    train.log
    data_manifest.json
    training_config.json
    step_<global-step>/
      model.safetensors
      config.json
      proprio_stats_sonic.json
      proprio_norm_config.json
```

### Stage defaults

| Stage | LR | Batch/process | Epochs | Stride | Warmup | Weight decay | Periodic checkpoint |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `stage1` | `1e-4` | 96 | 1 | 1 | 0.05 | 0.01 | disabled |
| `stage2` | `1e-4` | 96 | 1 | 1 | 0.05 | 0.01 | disabled |
| `stage3` | `3e-5` | 32 | 50,000-step cap | 1 | 0.1 | 0.005 | disabled; final only |

Shared defaults: `qwenpi_v3`, full Qwen training, bf16, DeepSpeed ZeRO-2,
16 action layers, 1024 action hidden size, 16 attention heads, 4 inference
steps, 8 workers, and no validation loop. Only the final checkpoint is saved
unless a periodic interval is explicitly supplied.

For explicit distributed settings:

```bash
bash scripts/train/train_huge.sh stage3 \
  --num-gpus 8 \
  --num-nodes 1 \
  --node-rank 0 \
  --master-addr 127.0.0.1 \
  --master-port 29500 \
  --init-ckpt none \
  --prop-roots /path/to/dataset \
  --proprio-stats-path /path/to/proprio_stats_union_sonic.json
```

Stage 1 and Stage 2 retain legacy `resource/dataset/` defaults when no explicit
roots are passed. Use repeated `--stage1-source`/`--stage2-source` arguments for
the current plural tree. See [Compatibility notes](#compatibility-notes).

## 📊 Offline evaluation

`offline_compare.py` plots predicted and ground-truth actions for proprietary
datasets:

```bash
MPLCONFIGDIR=/tmp/huge-matplotlib \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
PYTHONPATH=src \
.venv/bin/python scripts/eval/offline_compare.py \
  --ckpt '.runs/train/stage3/<run-id>/step_*' \
  --stage stage3 \
  --prop-roots /path/to/proprietary/dataset \
  --proprio-stats-path /path/to/proprio_stats_union_sonic.json \
  --device cuda \
  --batch-size 4 \
  --sample-stride 1 \
  --chunk-keep-steps 20 \
  --episode-idx 0 \
  --out-dir .runs/eval/offline_compare
```

Base mode writes one PNG per checkpoint. Add
`--visual-ablation --visual-swap-test` to produce visual-sensitivity JSON
diagnostics.

This entry point does not expose HumanoidArena manifest/task arguments and is
not a native HumanoidArena evaluator.

## 🌐 Inference server

```bash
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
PYTHONPATH=src \
.venv/bin/python src/huge/inference/run_huge_server.py \
  --ckpt .runs/train/stage3/<run-id>/step_<step> \
  --proprio-stats-path /path/to/proprio_stats_union_sonic.json \
  --host 127.0.0.1 \
  --port 5550 \
  --device cuda \
  --action-horizon 50 \
  --log-dir .runs/deploy_log/<run-id>
```

The server uses ZMQ REQ/REP with msgpack-numpy. It has no authentication or
TLS; do not expose it to an untrusted network.

`--log-dir` writes a JSONL manifest and per-call msgpack payloads under
`calls/`. The server first uses `--proprio-stats-path`, then a stats file
bundled in the checkpoint, then the original path in checkpoint metadata.

For standalone requests without the WBC state stream, add
`--no-state-tap-enable`.

## 🦾 WBC deployment

Real-robot deployment uses the independent WBC environment. Install
`GR00T-WholeBodyControl/.venv_inference` according to that project's
instructions. Run each process below in a separate terminal from the
repository root.

1. Start the Huge server above. Use `--host 0.0.0.0` only on a trusted
   control network.

2. Start the WBC deploy process:

   ```bash
   cd GR00T-WholeBodyControl/gear_sonic_deploy

   ./deploy.sh \
     --input-type zmq_manager \
     --zmq-host localhost \
     --hand-type inspire \
     real
   ```

3. Start the VLA client:

   ```bash
   cd GR00T-WholeBodyControl

   .venv_inference/bin/python gear_sonic/scripts/run_vla_inference.py \
     --host localhost \
     --port 5550 \
     --embodiment-tag unitree_g1_sonic \
     --prompt "Walk forward, grab the cola and throw it into the trash bin" \
     --action-publish-rate 50 \
     --action-horizon 50 \
     --camera-host 192.168.123.164 \
     --camera-port 5555
   ```

4. Start the keyboard publisher:

   ```bash
   cd GR00T-WholeBodyControl
   .venv_inference/bin/python keyboard_pub.py
   ```

The deploy process controls physical hardware and requires operator checks.
Do not automate confirmation or motor enablement.

## 🧪 HumanoidArena simulation evaluation

The wrapper uses HumanoidArena's Isaac Lab and LeRobot environments:

```bash
bash scripts/eval/humanoidarena/run_eval.sh \
  <task> <lerobot-model-search-root> [all|sonic|twist2]
```

Automatic discovery searches for `*/checkpoints/*/pretrained_model` and
two-level `<run>/pretrained_model` directories. Use `MODEL_PATHS_CSV` for an
explicit model path.

Supported tasks:

```text
boxing
doubledesk
football
open_door
pp_box
sit_sofa
vision_navi
```

One-model, one-episode smoke:

```bash
TEST_MODE=base_test \
SEEDS_OVERRIDE=0 \
REPEATS_PER_SEED=1 \
NUM_WORKERS=1 \
MODEL_LIMIT=1 \
MODEL_PATHS_CSV=/absolute/path/to/pretrained_model \
bash scripts/eval/humanoidarena/run_eval.sh \
  pp_box /absolute/path/to/existing/model-search-root sonic
```

Preview the evaluator command without launching simulation:

```bash
DRY_RUN=1 \
SKIP_PREFLIGHT=1 \
TEST_MODE=base_test \
SEEDS_OVERRIDE=0 \
REPEATS_PER_SEED=1 \
NUM_WORKERS=1 \
MODEL_LIMIT=1 \
MODEL_PATHS_CSV=/absolute/path/to/pretrained_model \
bash scripts/eval/humanoidarena/run_eval.sh \
  pp_box /absolute/path/to/existing/model-search-root sonic
```

The default full evaluation is not a smoke test. The preflight checks system
OpenGL libraries, TensorRT, cuDNN, and ONNX Runtime CUDA support.
`SKIP_PREFLIGHT=1` permits the run after a failed check; it does not suppress
the check output.

Base-test defaults are 3 seeds x 20 repeats: 60 episodes per model/backend.
The default `all` backend runs SONIC and TWIST2: 120 episodes per model.

This wrapper discovers HumanoidArena/LeRobot `pretrained_model` directories.
It does not load Huge `step_*/model.safetensors` checkpoints directly.

## ✅ Validation

HumanoidArena preprocessing tests:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
.venv/bin/python -m unittest discover -s tests \
  -p 'test_humanoid_arena_preprocess_*.py' -v
```

Python syntax:

```bash
PYTHONPYCACHEPREFIX=/tmp/huge-readme-pycache \
.venv/bin/python -m compileall -q src scripts tests
```

Shell syntax:

```bash
find scripts -type f -name '*.sh' -print0 | xargs -0 -n1 bash -n
```

## ⚠️ Compatibility notes

- Current data is stored under `resource/datasets/`. Stage 2, the general
  audit, and several legacy sim/real/preprocessing defaults still use
  `resource/dataset/`. Do not treat the paths as interchangeable.
- Stage 1 accepts `--egodex-root resource/datasets/EgoDex`; both Stage 1 and
  Stage 2 accept repeated typed source arguments for the plural tree.
- `scripts/download_humanoidarena_vision_navi.py` is not present. The
  ModelScope snippet above is the supported download command in this README.
- `tests/test_download_humanoidarena_vision_navi.py` still imports the removed
  downloader module; restore or remove that stale test before using full
  `test_*.py` discovery as a green check.
- `src/huge/data/audit.py` writes a report and still checks legacy singular
  paths; it is not a complete audit of the unified HumanoidArena cache.
- `offline_compare.py` is not a native HumanoidArena evaluator.
- The HumanoidArena simulation wrapper accepts LeRobot checkpoints, not native
  Huge checkpoints.

## 🙏 Acknowledgments

Huge builds on Psi0, GR00T-WholeBodyControl, Isaac-GR00T, GEAR-SONIC,
HumanoidArena, and Qwen3-VL.
