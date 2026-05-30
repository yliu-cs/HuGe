#!/usr/bin/env bash
set -euo pipefail

stage="${1:-stage2}"
shift || true

num_gpus=""
num_nodes=""
node_rank=""
master_addr=""
master_port=""
mixed_precision="${KAPPA_MIXED_PRECISION:-bf16}"
use_deepspeed="${KAPPA_USE_DEEPSPEED:-1}"
use_deepspeed_arg_seen=0
deepspeed_config="${KAPPA_DEEPSPEED_CONFIG-configs/deepspeed/ds_zero2.json}"
deepspeed_config_arg_seen=0
init_ckpt="${KAPPA_INIT_CKPT:-}"
init_ckpt_arg_seen=0
output_dir=".runs/kappa"
accelerate_config="${KAPPA_ACCELERATE_CONFIG:-}"
zero_stage=""
train_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --num_gpus|--num-gpus)
      num_gpus="$2"; shift 2 ;;
    --num_gpus=*|--num-gpus=*)
      num_gpus="${1#*=}"; shift ;;
    --num_nodes|--num-nodes|--num_machines|--num-machines)
      num_nodes="$2"; shift 2 ;;
    --num_nodes=*|--num-nodes=*|--num_machines=*|--num-machines=*)
      num_nodes="${1#*=}"; shift ;;
    --node_rank|--node-rank|--machine_rank|--machine-rank)
      node_rank="$2"; shift 2 ;;
    --node_rank=*|--node-rank=*|--machine_rank=*|--machine-rank=*)
      node_rank="${1#*=}"; shift ;;
    --master_addr|--master-addr|--main_process_ip|--main-process-ip)
      master_addr="$2"; shift 2 ;;
    --master_addr=*|--master-addr=*|--main_process_ip=*|--main-process-ip=*)
      master_addr="${1#*=}"; shift ;;
    --master_port|--master-port|--main_process_port|--main-process-port)
      master_port="$2"; shift 2 ;;
    --master_port=*|--master-port=*|--main_process_port=*|--main-process-port=*)
      master_port="${1#*=}"; shift ;;
    --mixed_precision|--mixed-precision)
      mixed_precision="$2"; shift 2 ;;
    --mixed_precision=*|--mixed-precision=*)
      mixed_precision="${1#*=}"; shift ;;
    --config_file|--config-file)
      accelerate_config="$2"; shift 2 ;;
    --config_file=*|--config-file=*)
      accelerate_config="${1#*=}"; shift ;;
    --use_deepspeed|--use-deepspeed)
      use_deepspeed=1; use_deepspeed_arg_seen=1; train_args+=("$1"); shift ;;
    --no-use_deepspeed|--no-use-deepspeed)
      use_deepspeed=0; use_deepspeed_arg_seen=1; train_args+=("$1"); shift ;;
    --deepspeed_config|--deepspeed-config)
      deepspeed_config="$2"; deepspeed_config_arg_seen=1; train_args+=("$1" "$2"); shift 2 ;;
    --deepspeed_config=*|--deepspeed-config=*)
      deepspeed_config="${1#*=}"; deepspeed_config_arg_seen=1; train_args+=("$1"); shift ;;
    --output_dir|--output-dir)
      output_dir="$2"; train_args+=("$1" "$2"); shift 2 ;;
    --output_dir=*|--output-dir=*)
      output_dir="${1#*=}"; train_args+=("$1"); shift ;;
    --init_ckpt|--init-ckpt)
      init_ckpt="$2"; init_ckpt_arg_seen=1; train_args+=("$1" "$2"); shift 2 ;;
    --init_ckpt=*|--init-ckpt=*)
      init_ckpt="${1#*=}"; init_ckpt_arg_seen=1; train_args+=("$1"); shift ;;
    --zero_stage|--zero-stage)
      zero_stage="$2"; train_args+=("$1" "$2"); shift 2 ;;
    --zero_stage=*|--zero-stage=*)
      zero_stage="${1#*=}"; train_args+=("$1"); shift ;;
    *)
      train_args+=("$1"); shift ;;
  esac
done

truthy() {
  case "${1,,}" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

previous_stage_for() {
  case "$1" in
    stage2|he_wbt) echo stage1 ;;
    stage3|prop|proprietary) echo stage2 ;;
    *) echo "" ;;
  esac
}

resolve_latest_stage_ckpt() {
  local stage_name="$1"
  local stage_root="$output_dir/$stage_name"
  [[ -d "$stage_root" ]] || return 1
  local latest_run
  latest_run="$(find "$stage_root" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -1)"
  [[ -n "$latest_run" ]] || return 1
  local latest_ckpt
  latest_ckpt="$(find "$latest_run" -mindepth 1 -maxdepth 1 -type d -name 'step_*' 2>/dev/null | sort -V | tail -1)"
  [[ -n "$latest_ckpt" ]] || return 1
  echo "$latest_ckpt"
}

count_visible_gpus() {
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != "NoDevFiles" ]]; then
    local cleaned="${CUDA_VISIBLE_DEVICES// /}"
    if [[ -n "$cleaned" ]]; then
      awk -F',' '{print NF}' <<<"$cleaned"
      return
    fi
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    local n
    n="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
    if [[ -n "$n" && "$n" -gt 0 ]]; then
      echo "$n"
      return
    fi
  fi
  echo 1
}

slurm_master_addr() {
  if [[ -n "${SLURM_NODELIST:-}" ]]; then
    if command -v scontrol >/dev/null 2>&1; then
      scontrol show hostnames "$SLURM_NODELIST" | head -n 1
      return
    fi
    echo "$SLURM_NODELIST" | sed 's/^\([^,[]*\).*/\1/'
    return
  fi
  echo ""
}

num_nodes="${num_nodes:-${KAPPA_NUM_NODES:-${PET_NNODES:-${SLURM_JOB_NUM_NODES:-1}}}}"
node_rank="${node_rank:-${KAPPA_NODE_RANK:-${PET_NODE_RANK:-${SLURM_NODEID:-${SLURM_PROCID:-0}}}}}"
master_addr="${master_addr:-${KAPPA_MASTER_ADDR:-${MASTER_ADDR:-$(slurm_master_addr)}}}"
master_addr="${master_addr:-127.0.0.1}"
master_port="${master_port:-${KAPPA_MASTER_PORT:-${MASTER_PORT:-29500}}}"
num_gpus="${num_gpus:-${KAPPA_GPUS_PER_NODE:-${KAPPA_NUM_GPUS:-$(count_visible_gpus)}}}"

if [[ -n "${KAPPA_NUM_PROCESSES:-}" ]]; then
  num_processes="$KAPPA_NUM_PROCESSES"
else
  num_processes=$((num_nodes * num_gpus))
fi
expected_num_processes="${KAPPA_EXPECTED_NUM_PROCESSES:-$num_processes}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_DEBUG="${NCCL_DEBUG:-ERROR}"

if [[ "$num_nodes" -gt 1 ]]; then
  export NCCL_IB_QPS_PER_CONNECTION="${NCCL_IB_QPS_PER_CONNECTION:-8}"
  export NCCL_GDR_LEVEL="${NCCL_GDR_LEVEL:-4}"
  export NCCL_IB_PCI_RELAXED_ORDERING="${NCCL_IB_PCI_RELAXED_ORDERING:-1}"
  export NCCL_IB_TC="${NCCL_IB_TC:-186}"
  export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
  export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
  export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-bond0}"
  export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond0}"
  export NCCL_IB_TIMEOUT="${NCCL_IB_TIMEOUT:-22}"
  export NCCL_IB_RETRY_CNT="${NCCL_IB_RETRY_CNT:-7}"
  export NCCL_IB_HCA="${NCCL_IB_HCA:-^=mlx5_3,mlx5_4,mlx5_5,mlx5_bond_0}"
  ulimit -l unlimited || true
fi

print_launch_config() {
  echo "==> Kappa launch: stage=$stage nodes=$num_nodes gpus_per_node=$num_gpus total_processes=$num_processes node_rank=$node_rank master=${master_addr}:${master_port} precision=$mixed_precision deepspeed=$use_deepspeed deepspeed_config=${deepspeed_config:-none}"
  if [[ -n "$init_ckpt" ]]; then
    echo "==> init_ckpt=$init_ckpt"
  fi
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "==> CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  fi
}

ensure_deepspeed_accelerate_config() {
  if [[ -n "$accelerate_config" ]]; then
    echo "$accelerate_config"
    return
  fi
  local config_dir="${KAPPA_ACCELERATE_CONFIG_DIR:-.runs/accelerate}"
  local config_path="$config_dir/deepspeed_local.yaml"
  mkdir -p "$config_dir"
  cat >"$config_path" <<EOF
compute_environment: LOCAL_MACHINE
distributed_type: DEEPSPEED
mixed_precision: "$mixed_precision"
use_cpu: false
debug: false
enable_cpu_affinity: false
EOF
  echo "$config_path"
}

launch_train() {
  local script="$1"
  shift
  local prev_stage
  local resolved_init_ckpt="$init_ckpt"
  if [[ "$init_ckpt_arg_seen" -eq 0 && -z "$resolved_init_ckpt" ]]; then
    prev_stage="$(previous_stage_for "$stage")"
    if [[ -n "$prev_stage" ]]; then
      resolved_init_ckpt="$(resolve_latest_stage_ckpt "$prev_stage")" || {
        echo "==> ERROR: no checkpoint found for previous stage '$prev_stage' under $output_dir" >&2
        echo "==>       run '$prev_stage' first, or pass --init-ckpt none to disable auto-init." >&2
        exit 1
      }
      init_ckpt="$resolved_init_ckpt"
    fi
  fi
  if [[ "$init_ckpt_arg_seen" -eq 0 && -n "$resolved_init_ckpt" ]]; then
    train_args+=(--init-ckpt "$resolved_init_ckpt")
  fi
  local launch_args=(
    launch
    --num_processes "$num_processes"
    --num_machines "$num_nodes"
    --machine_rank "$node_rank"
    --main_process_ip "$master_addr"
    --main_process_port "$master_port"
    --mixed_precision "$mixed_precision"
    --dynamo_backend no
  )
  if truthy "$use_deepspeed"; then
    launch_args+=(--config_file "$(ensure_deepspeed_accelerate_config)")
    launch_args+=(--use_deepspeed)
    if [[ -n "$deepspeed_config" ]]; then
      launch_args+=(--deepspeed_config_file "$deepspeed_config")
      if [[ "$deepspeed_config_arg_seen" -eq 0 ]]; then
        train_args+=(--deepspeed-config "$deepspeed_config")
      fi
    elif [[ -n "$zero_stage" ]]; then
      launch_args+=(--zero_stage "$zero_stage")
    fi
  elif [[ "$num_processes" -gt 1 ]]; then
    launch_args+=(--multi_gpu)
  elif [[ "$use_deepspeed_arg_seen" -eq 0 ]]; then
    train_args+=(--no-use-deepspeed)
  fi
  print_launch_config
  .venv/bin/accelerate "${launch_args[@]}" \
    "$script" \
    --use-accelerate \
    --expected-num-processes "$expected_num_processes" \
    --mixed-precision "$mixed_precision" \
    --use-qwen \
    "$@" \
    "${train_args[@]}"
}

case "$stage" in
  stage1|egodex)
    launch_train src/kappa/training/train_stage1_egodex.py "$@"
    ;;
  stage2|he_wbt)
    launch_train src/kappa/training/train_stage2_posttrain.py "$@"
    ;;
  stage3|prop|proprietary)
    launch_train src/kappa/training/train_stage3_finetune.py "$@"
    ;;
  audit)
    uv run python src/kappa/data/audit.py "${train_args[@]}"
    ;;
  *)
    echo "unknown stage: $stage" >&2
    exit 2
    ;;
esac
