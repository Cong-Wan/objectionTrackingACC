#!/usr/bin/env bash
# Author: wilbur
# Version: 1.0
# Date: 2026-08-06
# Description: 4 并发跑 trackCrop 批处理视频；默认 N=8、关闭 debug。
#   输入: /home/Datasets/UniformClassification/video
#   输出: /home/Datasets/UniformClassification/3588
#
# 用法:
#   ./scripts/runBatchTrackCrop.sh
#   ENGINE=/path/to.engine JOBS=4 N=8 ./scripts/runBatchTrackCrop.sh
#   ./scripts/runBatchTrackCrop.sh --engine /path/to.engine --jobs 4 --N 8

set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

VIDEO_DIR="${VIDEO_DIR:-/home/Datasets/UniformClassification/video}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/Datasets/UniformClassification/3588}"
ENGINE="${ENGINE:-/home/Disk-6T/Datasets/person_head_upbody/Weights/251121_v8mp6_1280_silu/weights/1batch_3090.engine}"
TRACKER_CONFIG="${TRACKER_CONFIG:-${PROJECT_ROOT}/config/tracker.yaml}"
BIN="${BIN:-${PROJECT_ROOT}/build/bin/trackCrop}"
N="${N:-8}"
JOBS="${JOBS:-4}"
CONF="${CONF:-0.01}"
IOA="${IOA:-0.5}"
MIN_SIZE="${MIN_SIZE:-70}"

usage() {
  cat <<EOF
Usage: $0 [options]

Options:
  --video-dir DIR     输入视频目录 (default: ${VIDEO_DIR})
  --output DIR        输出根目录 (default: ${OUTPUT_DIR})
  --engine PATH       TensorRT engine 路径
  --tracker-config P  tracker yaml
  --bin PATH          trackCrop 可执行文件
  --N INT             跳帧参数 (default: 8)
  --jobs INT          并发数 (default: 4)
  --conf FLOAT        预过滤 conf (default: 0.01)
  --ioa FLOAT         IoA 阈值 (default: 0.5)
  --min-size INT      最小裁剪边 (default: 70)
  -h, --help          帮助

环境变量同名也可覆盖: VIDEO_DIR OUTPUT_DIR ENGINE N JOBS ...
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --video-dir) VIDEO_DIR="$2"; shift 2 ;;
    --output) OUTPUT_DIR="$2"; shift 2 ;;
    --engine) ENGINE="$2"; shift 2 ;;
    --tracker-config) TRACKER_CONFIG="$2"; shift 2 ;;
    --bin) BIN="$2"; shift 2 ;;
    --N|--n) N="$2"; shift 2 ;;
    --jobs) JOBS="$2"; shift 2 ;;
    --conf) CONF="$2"; shift 2 ;;
    --ioa) IOA="$2"; shift 2 ;;
    --min-size) MIN_SIZE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ ! -x "${BIN}" ]]; then
  echo "[ERROR] trackCrop not found or not executable: ${BIN}" >&2
  echo "        build first: cmake --build ${PROJECT_ROOT}/build -j" >&2
  exit 1
fi
if [[ ! -f "${ENGINE}" ]]; then
  echo "[ERROR] engine not found: ${ENGINE}" >&2
  exit 1
fi
if [[ ! -d "${VIDEO_DIR}" ]]; then
  echo "[ERROR] video dir not found: ${VIDEO_DIR}" >&2
  exit 1
fi
if [[ ! -f "${TRACKER_CONFIG}" ]]; then
  echo "[ERROR] tracker config not found: ${TRACKER_CONFIG}" >&2
  exit 1
fi
if ! [[ "${N}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[ERROR] N must be positive int: ${N}" >&2
  exit 1
fi
if ! [[ "${JOBS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[ERROR] jobs must be positive int: ${JOBS}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"
LOG_DIR="${OUTPUT_DIR}/_logs"
mkdir -p "${LOG_DIR}"

mapfile -t VIDEOS < <(find "${VIDEO_DIR}" -maxdepth 1 -type f \
  \( -iname '*.mp4' -o -iname '*.mkv' -o -iname '*.avi' -o -iname '*.mov' -o -iname '*.ts' \) \
  | sort)

TOTAL=${#VIDEOS[@]}
if [[ "${TOTAL}" -eq 0 ]]; then
  echo "[ERROR] no videos under ${VIDEO_DIR}" >&2
  exit 1
fi

echo "========================================"
echo " trackCrop batch"
echo "  bin      : ${BIN}"
echo "  engine   : ${ENGINE}"
echo "  video_dir: ${VIDEO_DIR}"
echo "  output   : ${OUTPUT_DIR}"
echo "  tracker  : ${TRACKER_CONFIG}"
echo "  N=${N}  jobs=${JOBS}  conf=${CONF}  ioa=${IOA}  min-size=${MIN_SIZE}"
echo "  videos   : ${TOTAL}"
echo "  debug    : OFF"
echo "  logs     : ${LOG_DIR}"
echo "========================================"

run_one() {
  local video="$1"
  local stem
  stem="$(basename "${video}")"
  stem="${stem%.*}"
  local log_file="${LOG_DIR}/${stem}.log"
  local start_ts end_ts
  start_ts="$(date '+%F %T')"

  {
    echo "[$(date '+%F %T')] START ${video}"
    echo "cmd: ${BIN} --video ${video} --engine ${ENGINE} --N ${N} --output ${OUTPUT_DIR} --conf ${CONF} --ioa ${IOA} --min-size ${MIN_SIZE} --tracker-config ${TRACKER_CONFIG}"
    # 不开 --debug
    "${BIN}" \
      --video "${video}" \
      --engine "${ENGINE}" \
      --N "${N}" \
      --output "${OUTPUT_DIR}" \
      --conf "${CONF}" \
      --ioa "${IOA}" \
      --min-size "${MIN_SIZE}" \
      --tracker-config "${TRACKER_CONFIG}"
    local rc=$?
    end_ts="$(date '+%F %T')"
    echo "[${end_ts}] END rc=${rc} ${video} (start ${start_ts})"
    exit "${rc}"
  } >"${log_file}" 2>&1
}

# 简易并发池：最多 JOBS 个后台任务
declare -a PIDS=()
declare -a PID_VIDEO=()
FAIL=0
DONE=0
IDX=0

wait_one_slot() {
  # 等到有空位：轮询已完成的 pid
  while true; do
    local running=0
    local new_pids=()
    local new_vids=()
    local i
    for i in "${!PIDS[@]}"; do
      local pid="${PIDS[$i]}"
      if kill -0 "${pid}" 2>/dev/null; then
        new_pids+=("${pid}")
        new_vids+=("${PID_VIDEO[$i]}")
        running=$((running + 1))
      else
        local rc=0
        wait "${pid}" || rc=$?
        DONE=$((DONE + 1))
        if [[ "${rc}" -ne 0 ]]; then
          FAIL=$((FAIL + 1))
          echo "[FAIL] (${DONE}/${TOTAL}) ${PID_VIDEO[$i]} rc=${rc}  log=${LOG_DIR}/$(basename "${PID_VIDEO[$i]%.*}").log"
        else
          echo "[OK]   (${DONE}/${TOTAL}) ${PID_VIDEO[$i]}"
        fi
      fi
    done
    PIDS=("${new_pids[@]+"${new_pids[@]}"}")
    PID_VIDEO=("${new_vids[@]+"${new_vids[@]}"}")
    if [[ "${running}" -lt "${JOBS}" ]]; then
      break
    fi
    sleep 0.5
  done
}

for video in "${VIDEOS[@]}"; do
  wait_one_slot
  IDX=$((IDX + 1))
  echo "[LAUNCH ${IDX}/${TOTAL}] $(basename "${video}")"
  run_one "${video}" &
  PIDS+=("$!")
  PID_VIDEO+=("${video}")
done

# 收尾等待
while [[ "${#PIDS[@]}" -gt 0 ]]; do
  wait_one_slot
  sleep 0.2
done

echo "========================================"
echo " done: total=${TOTAL} ok=$((TOTAL - FAIL)) fail=${FAIL}"
echo " output: ${OUTPUT_DIR}"
echo " logs  : ${LOG_DIR}"
echo "========================================"

if [[ "${FAIL}" -gt 0 ]]; then
  exit 1
fi
exit 0
