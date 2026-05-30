#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

WATCH_DIR="${WATCH_DIR:-${REPO_ROOT}/.runs/train/stage3/20260530_103709}"
BAIDUPCS_BIN="${BAIDUPCS_BIN:-/inspire/qb-ilm2/project/robot-body/liujiacheng-240108120088/liuyang/Tools/BaiduPCS/BaiduPCS-Go}"
REMOTE_DIR="${BAIDU_REMOTE_DIR:-/Kappa/.runs/train/stage3}"
SESSION="${TMUX_SESSION:-baidu_sync_stage3_20260530_103709}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/.runs/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${SESSION}.log}"
LOCK_FILE="${LOCK_FILE:-${LOG_DIR}/${SESSION}.lock}"
STATE_FILE="${STATE_FILE:-${LOG_DIR}/${SESSION}.state}"

POLL_SECONDS="${POLL_SECONDS:-10}"
UPLOAD_THREADS="${UPLOAD_THREADS:-0}"
UPLOAD_CONCURRENCY="${UPLOAD_CONCURRENCY:-0}"
UPLOAD_RETRY="${UPLOAD_RETRY:-3}"

usage() {
  cat <<EOF
Usage:
  $0 [start] [remote_dir]
  $0 stop
  $0 status
  $0 attach
  $0 logs

Defaults:
  watch dir:  ${WATCH_DIR}
  remote dir: ${REMOTE_DIR}
  tmux:       ${SESSION}
  log:        ${LOG_FILE}

Environment overrides:
  WATCH_DIR BAIDUPCS_BIN BAIDU_REMOTE_DIR TMUX_SESSION LOG_DIR
  POLL_SECONDS STATE_FILE UPLOAD_THREADS UPLOAD_CONCURRENCY UPLOAD_RETRY
EOF
}

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

quote() {
  printf '%q' "$1"
}

file_list() {
  if [[ ! -d "${WATCH_DIR}" ]]; then
    return 2
  fi

  find "${WATCH_DIR}" -xdev -type f \
    ! -name '*.log' \
    ! -name '*.err.log' \
    ! -name '*.out.log' \
    ! -name '*.los' \
    -printf '%P\n' \
    | LC_ALL=C sort
}

ensure_remote_dir() {
  local dir="${1%/}"
  [[ -z "${dir}" || "${dir}" == "/" ]] && return 0

  local path=""
  local part
  IFS='/' read -r -a parts <<< "${dir#/}"
  for part in "${parts[@]}"; do
    [[ -z "${part}" ]] && continue
    path="${path}/${part}"
    "${BAIDUPCS_BIN}" mkdir "${path}" >/dev/null 2>&1 || true
  done
}

remote_root() {
  printf '%s/%s' "${REMOTE_DIR%/}" "$(basename -- "${WATCH_DIR}")"
}

upload_file() {
  local rel_path="$1"
  local local_path="${WATCH_DIR}/${rel_path}"
  local rel_dir
  local target_dir

  if [[ ! -f "${local_path}" ]]; then
    log "skip vanished file: ${rel_path}"
    return 0
  fi

  rel_dir="$(dirname -- "${rel_path}")"
  target_dir="$(remote_root)"
  if [[ "${rel_dir}" != "." ]]; then
    target_dir="${target_dir}/${rel_dir}"
  fi

  ensure_remote_dir "${target_dir}"
  log "upload file start: ${rel_path} -> ${target_dir}"
  "${BAIDUPCS_BIN}" upload \
    --policy rsync \
    --retry "${UPLOAD_RETRY}" \
    -p "${UPLOAD_THREADS}" \
    -l "${UPLOAD_CONCURRENCY}" \
    "${local_path}" \
    "${target_dir}"
  log "upload file done: ${rel_path}"
}

run_watcher() {
  mkdir -p "${LOG_DIR}"
  exec >>"${LOG_FILE}" 2>&1
  exec 9>"${LOCK_FILE}"
  if ! flock -n 9; then
    log "another watcher is already running, exit"
    exit 1
  fi

  if [[ ! -x "${BAIDUPCS_BIN}" ]]; then
    log "BaiduPCS-Go is not executable: ${BAIDUPCS_BIN}"
    exit 1
  fi

  log "watcher started"
  log "watch dir: ${WATCH_DIR}"
  log "remote root: $(remote_root)"
  log "poll=${POLL_SECONDS}s retry=${UPLOAD_RETRY}"

  local current_state
  local diff_state
  local rel_path
  local failed=0
  current_state="$(mktemp)"
  diff_state="$(mktemp)"
  trap 'rm -f "${current_state}" "${diff_state}"' EXIT
  mkdir -p "$(dirname -- "${STATE_FILE}")"
  if [[ ! -s "${STATE_FILE}" ]]; then
    file_list >"${STATE_FILE}" || true
    log "state baseline initialized from existing non-log files"
  fi

  while true; do
    if ! file_list >"${current_state}"; then
      log "watch dir does not exist yet: ${WATCH_DIR}"
      sleep "${POLL_SECONDS}"
      continue
    fi

    comm -13 "${STATE_FILE}" "${current_state}" >"${diff_state}" || true
    if [[ -s "${diff_state}" ]]; then
      log "new files detected, uploading immediately"
      failed=0
      while IFS= read -r rel_path; do
        if ! upload_file "${rel_path}"; then
          failed=1
          log "upload failed: ${rel_path}"
        fi
      done <"${diff_state}"

      if (( failed == 0 )); then
        cp "${current_state}" "${STATE_FILE}"
      else
        log "some uploads failed, will retry changed files on next loop"
      fi
    fi

    sleep "${POLL_SECONDS}"
  done
}

start_tmux() {
  local remote="${1:-${REMOTE_DIR}}"
  mkdir -p "${LOG_DIR}"

  if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "tmux session already running: ${SESSION}"
    echo "attach: $0 attach"
    echo "logs:   $0 logs"
    return 0
  fi

  local cmd
  cmd="cd $(quote "${REPO_ROOT}") && exec env"
  cmd+=" WATCH_DIR=$(quote "${WATCH_DIR}")"
  cmd+=" BAIDUPCS_BIN=$(quote "${BAIDUPCS_BIN}")"
  cmd+=" BAIDU_REMOTE_DIR=$(quote "${remote}")"
  cmd+=" TMUX_SESSION=$(quote "${SESSION}")"
  cmd+=" LOG_DIR=$(quote "${LOG_DIR}")"
  cmd+=" LOG_FILE=$(quote "${LOG_FILE}")"
  cmd+=" LOCK_FILE=$(quote "${LOCK_FILE}")"
  cmd+=" STATE_FILE=$(quote "${STATE_FILE}")"
  cmd+=" POLL_SECONDS=$(quote "${POLL_SECONDS}")"
  cmd+=" UPLOAD_THREADS=$(quote "${UPLOAD_THREADS}")"
  cmd+=" UPLOAD_CONCURRENCY=$(quote "${UPLOAD_CONCURRENCY}")"
  cmd+=" UPLOAD_RETRY=$(quote "${UPLOAD_RETRY}")"
  cmd+=" bash $(quote "$0") --watch"

  tmux new-session -d -s "${SESSION}" "${cmd}"
  echo "started tmux session: ${SESSION}"
  echo "watch dir: ${WATCH_DIR}"
  echo "remote dir: ${remote}"
  echo "logs: ${LOG_FILE}"
}

case "${1:-start}" in
  --watch)
    run_watcher
    ;;
  start)
    shift || true
    start_tmux "${1:-${REMOTE_DIR}}"
    ;;
  stop)
    tmux kill-session -t "${SESSION}"
    ;;
  status)
    tmux has-session -t "${SESSION}" 2>/dev/null \
      && echo "running: ${SESSION}" \
      || echo "not running: ${SESSION}"
    ;;
  attach)
    tmux attach -t "${SESSION}"
    ;;
  logs)
    mkdir -p "${LOG_DIR}"
    touch "${LOG_FILE}"
    tail -f "${LOG_FILE}"
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    start_tmux "${1}"
    ;;
esac
