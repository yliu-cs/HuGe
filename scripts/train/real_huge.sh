#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd -- "$REPO_ROOT"

dry_run=0
train_args=()
for arg in "$@"; do
  case "$arg" in
    --help|-h)
      printf '%s\n' \
        'Usage: bash scripts/train/real_huge.sh [--dry-run] [training options]' \
        '' \
        'Run only qwenpi_v3 R110_E400_syn (one training run).' \
        'Reuse the completed qwenpi_v3 R110_E0 baseline; it is not rerun.' \
        'Reuse the completed qwenpi_v3 R110_E400_raw experiment; it is not rerun.' \
        'Use balanced Real/Ego sampling with Ego probability 0.5.' \
        'Read contiguous episode blocks (64 samples by default), preserving source probabilities.' \
        'Initialize qwenpi_v3 from its latest published Stage 2 checkpoint:' \
        '  .runs/full_all_stage/all/qwenpi_v3/full/stage2/<run>/step_<N>' \
        'Latest means latest run name, then highest step; incomplete saves are excluded.' \
        'Check the initialization checkpoint before starting.' \
        'Default outputs: .runs/finetune_real/qwenpi_v3/full/stage3/<run>' \
        'Train for 20000 steps by default.' \
        'Save model checkpoints every 2000 steps; no periodic full resume states.' \
        'Training still saves a full resume state at completion.' \
        '' \
        'Use every CUDA GPU visible to this environment.' \
        'CUDA_VISIBLE_DEVICES is respected; busy GPUs are not excluded.' \
        'GPU/process counts are automatic; use CUDA_VISIBLE_DEVICES to limit GPUs.' \
        'This script does not stop GrabGPU or any other process.' \
        '--dry-run detects GPUs and prints the command without starting training.' \
        '' \
        'Use the frozen Stage 1/2 + Real/Ego union stats.' \
        'The default training seed is 7; episode selection follows it unless overridden.' \
        'Architecture, initialization, episode counts, video source and mixing are fixed.' \
        'For custom initialization/resume, use train_huge.sh stage3.' \
        'Other options (e.g. --max-steps, --seed) override the training defaults.' \
        'Per-device batch and accumulation are not rescaled for GPU count.'
      exit 0
      ;;
    --dry-run) dry_run=1 ;;
    *)
      option="${arg%%=*}"
      case "${option//_/-}" in
        --stage|--model-architecture|--init-ckpt|--resume-from-checkpoint|--real-num-episodes|--ego-num-episodes|--ego-video-source|--mixture-mode|--ego-sampling-prob|--humanoid-arena-roots)
          echo "ERROR: $option is fixed by this experiment; use train_huge.sh for a custom run." >&2
          exit 2
          ;;
      esac
      train_args+=("$arg")
      ;;
  esac
done

# Ask the training environment, not nvidia-smi's host-wide device inventory.
num_gpus="$(.venv/bin/python -c 'import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)')"
if [[ "$num_gpus" -lt 1 ]]; then
  echo 'ERROR: no CUDA GPUs are available in this environment; check CUDA_VISIBLE_DEVICES and the driver.' >&2
  exit 1
fi

architectures=(qwenpi_v3)
experiments=(
  R110_E400_syn
)
checkpoints=()
for architecture in "${architectures[@]}"; do
  stage2_root=".runs/full_all_stage/all/$architecture/full/stage2"
  if [[ ! -d "$stage2_root" ]]; then
    echo "ERROR: Stage 2 directory not found: $stage2_root" >&2
    exit 1
  fi
  # Match only published portable checkpoints, never _resume_states or staging.
  checkpoint="$(find "$stage2_root" -mindepth 3 -maxdepth 3 -type f \
    -path "$stage2_root/*/step_[0-9]*/_SUCCESS" -printf '%h\n' | sort -V | tail -n 1)"
  if [[ -z "$checkpoint" || ! -s "$checkpoint/config.json" || ! -s "$checkpoint/model.safetensors" || ! -s "$checkpoint/manifest.json" ]]; then
    echo "ERROR: no complete published Stage 2 checkpoint under $stage2_root" >&2
    exit 1
  fi
  checkpoints+=("$checkpoint")
  echo "==> $architecture Stage 2 initialization: $checkpoint"
done

# The existing launcher owns Accelerate, DeepSpeed, cuDNN and node topology.
# Recompute world size from nodes times detected GPUs, ignoring stale overrides.
unset HUGE_NUM_PROCESSES HUGE_EXPECTED_NUM_PROCESSES HUGE_RESUME_FROM_CHECKPOINT
echo "==> Detected $num_gpus CUDA GPUs (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-<unset>})"
for experiment in "${experiments[@]}"; do
  IFS=_ read -r real_count ego_count video_source <<<"$experiment"
  for i in "${!architectures[@]}"; do
    architecture="${architectures[$i]}"
    echo "==> Stage 3 experiment: $architecture $experiment"
    cmd=(
      bash "$SCRIPT_DIR/train_huge.sh" stage3
      --mixed-precision bf16 --use-deepspeed
      --deepspeed-config configs/deepspeed/ds_zero2.json
      --no-freeze-qwen --no-qwen-lora --no-train-qwen-lora
      --vlm-learning-rate 1e-5 --projector-learning-rate 2.5e-5 --action-learning-rate 1e-4
      --min-lr-rate 0.04
      --max-steps 20000
      --checkpointing-steps 2000 --resume-state-steps 0
      --dataset-root resource/datasets
      --prop-roots resource/datasets/proprietary/real/cola_final_110
      --prop-ego-roots resource/datasets/proprietary/ego
      --proprio-stats-path resource/datasets/meta/stage123_proprietary_proprio_stats_union_sonic.json
      --action-hidden-size 1024 --num-action-layers 16 --num-attention-heads 16
      --output-dir ".runs/finetune_real/$architecture/full"
      "${train_args[@]}"
      --model-architecture "$architecture" --init-ckpt "${checkpoints[$i]}"
      --real-num-episodes "${real_count#R}" --ego-num-episodes "${ego_count#E}"
      --ego-video-source "$video_source" --mixture-mode balanced --ego-sampling-prob 0.5
      --num-gpus "$num_gpus"
    )
    if [[ "$dry_run" -eq 1 ]]; then
      printf '%q ' "${cmd[@]}"
      printf '\n'
    else
      "${cmd[@]}"
    fi
  done
done
